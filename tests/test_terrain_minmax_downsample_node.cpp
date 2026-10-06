#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
#include <cstring>
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainMinMaxDownsampleNode.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"

using namespace brassica;

// Stage 4's two proofs, kept in one file since they're about the same pair of nodes:
//  1. The schedule/barrier proof -- pure CPU-side graph logic, no device needed. Confirms
//     Graph::CollectEdges's ordinary producer/consumer handling actually orders
//     TerrainMinMaxDownsampleNode after TerrainGenNode and that the frame graph itself
//     synthesizes a real barrier for TerrainMinMaxTexture at the stage boundary -- neither node
//     writes a manual vkCmd.pipelineBarrier2 anywhere.
//  2. The real-GPU correctness proof -- an actual dispatch of a deterministic mip-0 writer feeding
//     the real TerrainMinMaxDownsampleNode, reading back real mip-0 and mip-1 texels and checking
//     mip 1 is the true min/max of mip 0's matching 2x2 footprint.

TEST_CASE(
	"TerrainMinMaxDownsampleNode lands in a later stage than TerrainGenNode, with a real "
	"synthesized barrier for TerrainMinMaxTexture at the boundary"
) {
	TerrainGenNode              genNode;
	TerrainMinMaxDownsampleNode downsampleNode;

	graph::Graph g;
	// genNode Creates<TerrainMinMaxTexture>, downsampleNode Modifies<TerrainMinMaxTexture> --
	// an ordinary producer/consumer edge (downsampleNode's Consumes names genNode's Produces),
	// which Graph::CollectEdges orders the normal way. Registration order doesn't drive this
	// (unlike a same-phase self-modify chain); it's listed first here only to match
	// AllNodes.hpp's #include order for readability.
	g.RegisterRef(genNode);
	g.RegisterRef(downsampleNode);

	graph::FrameContext ctx{.width = 1024, .height = 1024};
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

	std::optional<std::size_t> genStage = stageOf(0);
	std::optional<std::size_t> downsampleStage = stageOf(1);
	REQUIRE(genStage.has_value());
	REQUIRE(downsampleStage.has_value());

	// The real proof the edge was collected at all: a missing producer/consumer edge would still
	// compile (nothing requires downsampleStage > genStage structurally), just with no barrier
	// between them -- a silent, not a loud, failure mode.
	CHECK(*downsampleStage > *genStage);

	const graph::ResourceId minMaxId = graph::IdOf<TerrainMinMaxTexture>();
	bool                    foundSynthesizedBarrier = false;
	for (std::size_t s = *genStage + 1; s <= *downsampleStage; ++s) {
		for (const auto& b : schedule.stages[s].preBarriers.Items()) {
			if (b.resource == minMaxId) {
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

	// Deliberately not the real TerrainGenNode: that node also unconditionally compiles
	// terrain_aabb.comp (declares the PhysicalStorageBufferAddresses SPIR-V capability --
	// bufferDeviceAddress isn't a feature MinimalDevice enables) and Modifies<TerrainTLAS> (whose
	// self-modify barrier derives ACCELERATION_STRUCTURE_BUILD_BIT_KHR regardless of whether the
	// device has the accelerationStructure feature). Both are pre-existing gaps in running
	// TerrainGenNode against a device without mesh-shader/ray-query support, orthogonal to
	// TerrainMinMaxDownsampleNode -- flagged separately, not fixed here. This node writes mip 0 of
	// TerrainMinMaxTexture with a simple, exactly predictable per-texel function instead, so the
	// correctness check below can compare against hand-computed values rather than an opaque
	// analytical terrain function's output.
	struct FakeHeightWriterNode {
		using Resources = graph::Declares<graph::Modify<TerrainMinMaxTexture>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            shader;
		glm::uvec4               gridParams{14, 16, 3584, 1024}; // numLODs, _, _, mapDim

		static constexpr const char* kShaderSource = R"(
			#version 460
			#extension GL_EXT_nonuniform_qualifier : require
			layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
			layout(set = 1, binding = 3) writeonly uniform image2DArray uImageArraysGenericWrite[];
			layout(push_constant) uniform PC { uint outIdx; } pc;
			void main() {
				ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
				float h = float(coord.x) + float(coord.y) * 0.01;
				imageStore(uImageArraysGenericWrite[nonuniformEXT(pc.outIdx)], coord, vec4(h, h, 0.0, 0.0));
			}
		)";

		void Init(vk::Device device, render::PipelineLibrary* lib) {
			pipelineLibrary = lib;
			shader.CompileFromSource(device, kShaderSource, shaderc_glsl_compute_shader, "fake_height_writer");
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxDesc(gridParams.x),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			struct PushConstants {
				std::uint32_t outIdx;
			} push{.outIdx = ctx.StorageIndex<TerrainMinMaxTexture>()};

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)),
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet)),
			};

			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);

			// Covers the full 1024x1024xnumLODs image -- every texel SPD's reduction touches gets
			// a real, defined value, not just the (0,0) corner this test actually reads back.
			vkCmd.dispatch(gridParams.w / 16, gridParams.w / 16, gridParams.x);
		}
	};

	// Real frame set (set 0, zeroed FrameUBO -- forceRegeneration bypasses every camera-delta
	// check in terrain_gen.comp, so the zeroed contents never matter) + bindless set (set 1) with
	// everything both nodes' shaders statically reference: sampled 2D (unused but declared),
	// sampled 2D array (terrain_gen.comp/terrain_downsample.comp's inputs), a real sampler at
	// BRASSICA_SAMPLER_NEAREST_CLAMP's index, and a storage-image binding sized for the terrain
	// quartet's indices (TerrainClipmapTexture/TerrainBiomeTexture single indices + TerrainMinMax
	// Texture's 11 per-mip indices).
	struct TerrainDownsampleBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                       sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	TerrainDownsampleBindlessSet CreateTerrainDownsampleBindlessSet(vk::Device device, VmaAllocator allocator) {
		TerrainDownsampleBindlessSet result;

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
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		layoutBindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
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
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 16},
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

	void DestroyTerrainDownsampleBindlessSet(
		vk::Device                     device,
		VmaAllocator                    allocator,
		TerrainDownsampleBindlessSet& s
	) {
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

TEST_CASE(
	"TerrainMinMaxDownsampleNode produces a real, correct min/max mip chain on real hardware"
) {
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
		TerrainDownsampleBindlessSet    bindlessSet = CreateTerrainDownsampleBindlessSet(vkDevice, allocator);
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);

		graph::PhysicalExecutionBackend backend(registry);

		FakeHeightWriterNode fakeWriter;
		fakeWriter.Init(vkDevice, &pipelineLibrary);

		TerrainMinMaxDownsampleNode downsampleNode;
		downsampleNode.Init(
			render::NodeServices{
				.device = vkDevice,
				.pipelineLibrary = &pipelineLibrary,
				.physicalRegistry = &registry,
			}
		);
		Shader::ClearConstants();

		graph::Graph g;
		g.RegisterRef(fakeWriter);
		g.RegisterRef(downsampleNode);

		graph::FrameContext ctx{.width = 1024, .height = 1024};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(g, ctx, cmd, false));

		auto tex = registry.GetTexture<TerrainMinMaxTexture>();
		REQUIRE(tex != nullptr);
		vk::Image image = tex->GetImage();

		// Compute-write -> transfer-read, staying in General throughout (a legal source layout
		// for vkCmdCopyImageToBuffer), covering every mip the dispatch touched.
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

		// mip 0's 2x2 footprint at (0,0)-(1,1), LOD slice 0 -- ground truth for mip 1's (0,0)
		// texel, read back as real RG32F data (4 texels x 2 floats x 4 bytes = 32 bytes).
		constexpr vk::DeviceSize kMip0Bytes = 4 * 2 * sizeof(float);
		constexpr vk::DeviceSize kMip1Bytes = 1 * 2 * sizeof(float);

		vk::BufferCreateInfo readbackBufferInfo{};
		readbackBufferInfo.setSize(kMip0Bytes + kMip1Bytes).setUsage(vk::BufferUsageFlagBits::eTransferDst);
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

		vk::BufferImageCopy mip0Copy{};
		mip0Copy.setBufferOffset(0)
			.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
			.setImageOffset(vk::Offset3D{0, 0, 0})
			.setImageExtent(vk::Extent3D{2, 2, 1});
		vkCmd.copyImageToBuffer(image, vk::ImageLayout::eGeneral, readbackBuffer, mip0Copy);

		vk::BufferImageCopy mip1Copy{};
		mip1Copy.setBufferOffset(kMip0Bytes)
			.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 1, 0, 1})
			.setImageOffset(vk::Offset3D{0, 0, 0})
			.setImageExtent(vk::Extent3D{1, 1, 1});
		vkCmd.copyImageToBuffer(image, vk::ImageLayout::eGeneral, readbackBuffer, mip1Copy);

		vkCmd.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		std::array<float, 8> mip0{}; // 4 texels x (min, max)
		std::array<float, 2> mip1{};
		std::memcpy(mip0.data(), readbackAllocResultInfo.pMappedData, kMip0Bytes);
		std::memcpy(
			mip1.data(),
			static_cast<const std::byte*>(readbackAllocResultInfo.pMappedData) + kMip0Bytes,
			kMip1Bytes
		);

		// FakeHeightWriterNode wrote (h, h) per texel with h = x + y*0.01 -- min == max there, so
		// each texel's own min/max is just mip0[i*2]. For (0,0),(1,0),(0,1),(1,1): h = 0, 1, 0.01,
		// 1.01 -- hand-computable, unlike an analytical terrain function's output.
		float expectedMin = std::min({mip0[0], mip0[2], mip0[4], mip0[6]});
		float expectedMax = std::max({mip0[1], mip0[3], mip0[5], mip0[7]});
		CHECK(expectedMin == doctest::Approx(0.0f));
		CHECK(expectedMax == doctest::Approx(1.01f));

		CHECK(mip1[0] == doctest::Approx(expectedMin));
		CHECK(mip1[1] == doctest::Approx(expectedMax));

		pipelineLibrary.Reset();
		fakeWriter.Destroy(vkDevice);
		downsampleNode.Destroy(vkDevice);
		vmaDestroyBuffer(allocator, rawReadbackBuffer, readbackAllocation);
		DestroyTerrainDownsampleBindlessSet(vkDevice, allocator, bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
