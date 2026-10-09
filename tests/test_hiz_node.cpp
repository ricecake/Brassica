#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <algorithm>
#include <array>
#include <cstring>
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/HiZDownsampleNode.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"

using namespace brassica;

// Seeds GBufferDepth with a known pattern via a direct, manually-recorded buffer->image copy,
// not ctx.WriteSpan<K> (the usual Staged-host-write convenience path): PhysicalRegistry::
// Provision collapses every node's realization of a shared key into a single map entry keyed
// only by ResourceId (`realizationsToProvision[resolvedKey] = r`), keeping whichever node's
// realization is visited *last* in schedule order. HiZDownsampleNode's own plain Read<GBufferDepth>
// (hostAccess=None, scheduled after this writer) always wins that overwrite, so a hostAccess=
// Staged override on just this node's own desc never actually reaches the stored PhysicalTexture
// -- BeginHostWrite reads hostAccess=None off it and silently no-ops. This is harmless in real
// production graphs (every real touch of a shared key already uses a byte-identical desc, so it
// never matters which one "wins"); not something to fix here, since nothing but this test
// currently gives one key's realizations different hostAccess values. Also: the real
// DepthBufferDesc never grants eStorage usage (depth-format storage-image support isn't part of
// Vulkan's mandatory format-feature set), so a compute-shader imageStore approach doesn't work
// either -- it would resolve to the bindless-miss fallback index and silently write nothing.
struct FakeDepthWriterNode {
	using Resources = graph::Declares<graph::Create<GBufferDepth>>;

	VmaAllocator                     allocator{nullptr};
	graph::PhysicalResourceRegistry* registry{nullptr};
	vk::Buffer                       stagingBuffer{nullptr};
	VmaAllocation                    stagingAllocation{nullptr};

	void Init(VmaAllocator alloc, graph::PhysicalResourceRegistry* reg) {
		allocator = alloc;
		registry = reg;
	}

	void Destroy(vk::Device) {
		if (stagingBuffer && allocator) {
			vmaDestroyBuffer(allocator, stagingBuffer, stagingAllocation);
			stagingBuffer = nullptr;
			stagingAllocation = nullptr;
		}
	}

	graph::Recipe Setup(const graph::FrameContext& ctx) {
		graph::Recipe r{.domain = graph::ExecutionDomain::Transfer};
		r.realizations.push_back(
			graph::ResourceRealization{
				.key = graph::IdOf<GBufferDepth>(),
				.access = graph::AccessKind::Write,
				.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
			}
		);
		return r;
	}

	void Execute(graph::NodeContext& ctx) {
		if (!allocator || !registry) {
			return;
		}
		auto tex = registry->GetTexture<GBufferDepth>();
		if (!tex) {
			return;
		}

		if (!stagingBuffer) {
			vk::DeviceSize bufferSize = vk::DeviceSize(ctx.width) * ctx.height * sizeof(float);

			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = bufferSize;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

			VmaAllocationCreateInfo allocCreateInfo{};
			allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocCreateInfo.flags =
				VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          rawBuffer{};
			VmaAllocation     rawAllocation{};
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &rawBuffer, &rawAllocation, &allocResultInfo) !=
			    VK_SUCCESS) {
				return;
			}
			stagingBuffer = rawBuffer;
			stagingAllocation = rawAllocation;

			auto* dst = static_cast<float*>(allocResultInfo.pMappedData);
			for (std::uint32_t y = 0; y < ctx.height; ++y) {
				for (std::uint32_t x = 0; x < ctx.width; ++x) {
					dst[y * ctx.width + x] = 1.0f - (float(x) * 0.001f + float(y) * 0.002f);
				}
			}
		}

		vk::CommandBuffer   vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
		vk::BufferImageCopy region{};
		region.setBufferOffset(0)
			.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eDepth, 0, 0, 1})
			.setImageOffset(vk::Offset3D{0, 0, 0})
			.setImageExtent(vk::Extent3D{ctx.width, ctx.height, 1});
		vkCmd.copyBufferToImage(stagingBuffer, tex->GetImage(), tex->GetCurrentLayout(), region);
	}
};

TEST_CASE("HiZDownsampleNode graph ordering and schedule proof") {
	FakeDepthWriterNode depthWriter;
	HiZDownsampleNode   hizNode;

	graph::Graph g;
	g.RegisterRef(depthWriter);
	g.RegisterRef(hizNode);

	graph::FrameContext ctx{.width = 1280, .height = 720};
	g.Setup(ctx);
	REQUIRE(g.Compile().has_value());

	const auto& schedule = g.GetSchedule();

	auto stageOf = [&](std::size_t nodeIndex) -> std::optional<std::size_t> {
		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			for (std::size_t n : schedule.stages[s].nodes) {
				if (n == nodeIndex) {
					return s;
				}
			}
		}
		return std::nullopt;
	};

	std::optional<std::size_t> writerStage = stageOf(0);
	std::optional<std::size_t> hizStage = stageOf(1);
	REQUIRE(writerStage.has_value());
	REQUIRE(hizStage.has_value());

	CHECK(*hizStage > *writerStage);

	const graph::ResourceId depthId = graph::IdOf<GBufferDepth>();
	bool                    foundSynthesizedBarrier = false;
	for (std::size_t s = *writerStage + 1; s <= *hizStage; ++s) {
		for (const auto& b : schedule.stages[s].preBarriers.Items()) {
			if (b.resource == depthId) {
				foundSynthesizedBarrier = true;
			}
		}
	}
	CHECK(foundSynthesizedBarrier);
}

namespace {

	void RegisterBindlessSamplerConstants() {
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);
	}

	struct HiZBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                       sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	HiZBindlessSet CreateHiZBindlessSet(vk::Device device, VmaAllocator allocator) {
		HiZBindlessSet result;

		vk::DescriptorSetLayoutBinding uboBinding{};
		uboBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(uboBinding);
		result.frameLayout = device.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize       framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize).setMaxSets(1);
		result.framePool = device.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(result.framePool).setSetLayouts(result.frameLayout);
		vk::DescriptorSet frameSet = device.allocateDescriptorSets(frameAllocInfo).front();

		VkBufferCreateInfo frameBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		frameBufferInfo.size = sizeof(FrameUBO);
		frameBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VmaAllocationCreateInfo frameAllocCreateInfo{};
		frameAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		frameAllocCreateInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
		VkBuffer          frameBuffer = VK_NULL_HANDLE;
		VmaAllocationInfo frameAllocResultInfo{};
		vmaCreateBuffer(
			allocator,
			&frameBufferInfo,
			&frameAllocCreateInfo,
			&frameBuffer,
			&result.frameUboAllocation,
			&frameAllocResultInfo
		);
		result.frameUboBuffer = frameBuffer;
		if (frameAllocResultInfo.pMappedData) {
			std::memset(frameAllocResultInfo.pMappedData, 0, sizeof(FrameUBO));
		}

		vk::DescriptorBufferInfo frameBufferDescInfo{result.frameUboBuffer, 0, sizeof(FrameUBO)};
		vk::WriteDescriptorSet   uboWrite{};
		uboWrite.setDstSet(frameSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eUniformBuffer);
		uboWrite.setBufferInfo(frameBufferDescInfo);
		device.updateDescriptorSets(uboWrite, {});

		std::array<vk::DescriptorSetLayoutBinding, 4> layoutBindings{};
		layoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(16)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[2]
			.setBinding(2)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(4)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[3]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eStorageImage)
			.setDescriptorCount(64)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 4> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(layoutBindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 3> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 32},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4},
			vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 64},
		};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSizes).setMaxSets(1).setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		result.pool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(result.pool).setSetLayouts(result.layout);
		vk::DescriptorSet set = device.allocateDescriptorSets(allocInfo).front();

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		result.sampler = device.createSampler(samplerInfo);

		vk::DescriptorImageInfo samplerImageInfo{};
		samplerImageInfo.setSampler(result.sampler);
		vk::WriteDescriptorSet samplerWrite{};
		samplerWrite.setDstSet(set).setDstBinding(2).setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfo);
		device.updateDescriptorSets(samplerWrite, {});

		result.bindings.set = set;
		result.bindings.layout = result.layout;
		result.bindings.sampledImage2DBinding = 0;
		result.bindings.sampledImage2DArrayBinding = 1;
		result.bindings.samplerBinding = 2;
		result.bindings.storageImageBinding = 3;
		result.bindings.frameSet = frameSet;
		result.bindings.frameSetLayout = result.frameLayout;
		return result;
	}

	void DestroyHiZBindlessSet(vk::Device device, VmaAllocator allocator, HiZBindlessSet& s) {
		device.destroySampler(s.sampler);
		device.destroyDescriptorPool(s.pool);
		device.destroyDescriptorSetLayout(s.layout);
		if (s.frameUboBuffer && s.frameUboAllocation) {
			vmaDestroyBuffer(allocator, s.frameUboBuffer, s.frameUboAllocation);
		}
		device.destroyDescriptorPool(s.framePool);
		device.destroyDescriptorSetLayout(s.frameLayout);
	}

} // namespace

TEST_CASE("HiZDownsampleNode GPU execution test") {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device   vkDevice = device.GetDevice();
	VmaAllocator allocator = device.GetAllocator();

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				device.GetQueueFamily(),
			}
		);
		vk::CommandBuffer vkCmd =
			vkDevice.allocateCommandBuffers(vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1})
				.front();

		RegisterBindlessSamplerConstants();

		render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);

		graph::PhysicalResourceRegistry registry(vkDevice, allocator);
		HiZBindlessSet                  bindlessSet = CreateHiZBindlessSet(vkDevice, allocator);
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);

		graph::PhysicalExecutionBackend backend(registry);

		FakeDepthWriterNode depthWriter;
		depthWriter.Init(allocator, &registry);

		HiZDownsampleNode hizNode;
		hizNode.Init(
			render::NodeServices{
				.device = vkDevice,
				.pipelineLibrary = &pipelineLibrary,
				.physicalRegistry = &registry,
			}
		);
		Shader::ClearConstants();

		graph::Graph g;
		g.RegisterRef(depthWriter);
		g.RegisterRef(hizNode);

		graph::FrameContext ctx{.width = 1024, .height = 1024};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));

		auto tex = registry.GetTexture<HiZTexture>();
		REQUIRE(tex != nullptr);
		vk::Image image = tex->GetImage();

		vk::ImageMemoryBarrier2 toTransferRead{};
		toTransferRead.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
			.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
			.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer)
			.setDstAccessMask(vk::AccessFlagBits2::eTransferRead)
			.setOldLayout(vk::ImageLayout::eGeneral)
			.setNewLayout(vk::ImageLayout::eGeneral)
			.setImage(image)
			.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 11, 0, 1});
		vk::DependencyInfo toTransferReadDep{};
		toTransferReadDep.setImageMemoryBarriers(toTransferRead);
		vkCmd.pipelineBarrier2(toTransferReadDep);

		constexpr vk::DeviceSize kMip1Bytes = 1 * 2 * sizeof(float);

		vk::BufferCreateInfo readbackBufferInfo{};
		readbackBufferInfo.setSize(kMip1Bytes).setUsage(vk::BufferUsageFlagBits::eTransferDst);
		VkBufferCreateInfo rawReadbackInfo = static_cast<VkBufferCreateInfo>(readbackBufferInfo);

		VmaAllocationCreateInfo readbackAllocInfo{};
		readbackAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		readbackAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          rawReadbackBuffer{};
		VmaAllocation     readbackAllocation{};
		VmaAllocationInfo readbackAllocResultInfo{};
		REQUIRE(
			vmaCreateBuffer(
				allocator,
				&rawReadbackInfo,
				&readbackAllocInfo,
				&rawReadbackBuffer,
				&readbackAllocation,
				&readbackAllocResultInfo
			) == VK_SUCCESS
		);
		vk::Buffer readbackBuffer{rawReadbackBuffer};

		vk::BufferImageCopy mip1Copy{};
		mip1Copy.setBufferOffset(0)
			.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 1, 0, 1})
			.setImageOffset(vk::Offset3D{0, 0, 0})
			.setImageExtent(vk::Extent3D{1, 1, 1});
		vkCmd.copyImageToBuffer(image, vk::ImageLayout::eGeneral, readbackBuffer, mip1Copy);

		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		std::array<float, 2> mip1{}; // min, max
		std::memcpy(mip1.data(), readbackAllocResultInfo.pMappedData, kMip1Bytes);

		// FakeDepthWriter wrote d = 1.0 - (coord.x*0.001 + coord.y*0.002).
		// For (0,0),(1,0),(0,1),(1,1):
		// d(0,0) = 1.0
		// d(1,0) = 0.999
		// d(0,1) = 0.998
		// d(1,1) = 0.997
		// Expected min = 0.997, max = 1.0
		CHECK(mip1[0] == doctest::Approx(0.997f));
		CHECK(mip1[1] == doctest::Approx(1.0f));

		pipelineLibrary.Reset();
		depthWriter.Destroy(vkDevice);
		hizNode.Destroy(vkDevice);
		vmaDestroyBuffer(allocator, rawReadbackBuffer, readbackAllocation);
		DestroyHiZBindlessSet(vkDevice, allocator, bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
