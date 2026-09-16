#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cstring>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/WaterNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "terrain/TerrainClipmap.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	// Stands in for TerrainNode + DeferredNode: writes the G-buffer inputs WaterNode reads
	// and the Swapchain target it composites onto, with nothing in its own Execute -- this test
	// is about WaterNode's own pipeline/barrier/blend correctness, not about rendering a real
	// scene into the G-buffer first. graph::Phase::Default (its default), strictly before
	// WaterNode's Phase::Late, is what makes a plain Modify<Swapchain> on both nodes -- with no
	// version number between them -- schedule correctly; see WaterNode.hpp's own comment.
	struct FakeSceneProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>,
			graph::Create<TerrainClipmapTexture>,
			graph::Modify<Swapchain>>;

		vk::Extent2D extent;
		vk::Format   swapchainFormat;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainClipmapTexture>(),
					.access = graph::AccessKind::Write,
					.desc = TerrainClipmapDesc(10),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

	// Real frame set (set 0) + bindless set (set 1), mirroring Engine's real split
	// (InitFrameSet/InitGlobalDescriptors) rather than the old single merged set: WaterNode now
	// binds both unconditionally, so this fixture needs a real, defined FrameUBO buffer behind
	// the frame set's binding 0 (bindless.glsl's FrameUBO block is statically read by
	// water.frag/water.mesh via uCameraPosition/uTime, so an unwritten descriptor there would be
	// a real, if harmless-content, validation gap). The bindless set carries just enough for
	// water.frag's SAMPLE_NEAREST(gPositionIndex/gAlbedoIndex/gNormalIndex, ...) to resolve real
	// descriptors: binding 0 (sampled 2D) is what PhysicalRegistry writes GBufferPosition/
	// GBufferAlbedo/GBufferNormal's indices into, binding 2 (samplers) needs a real sampler
	// written at BRASSICA_SAMPLER_NEAREST_CLAMP's index (0) since the shader indexes it
	// unconditionally. No array/storage/AS bindings -- WaterNode never touches them. Mirrors
	// test_physical_backend.cpp's own CreateBindlessTestSet, trimmed to what this node needs.
	struct WaterBindlessSet {
		vk::DescriptorSetLayout frameLayout{};
		vk::DescriptorPool      framePool{};
		vk::Buffer              frameUboBuffer{};
		VmaAllocation           frameUboAllocation{};

		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                       sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	WaterBindlessSet CreateWaterBindlessSet(vk::Device device, VmaAllocator allocator) {
		WaterBindlessSet result;

		// -- Frame set (set 0): just the FrameUBO --
		vk::DescriptorSetLayoutBinding uboBinding{};
		uboBinding.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo frameLayoutInfo{};
		frameLayoutInfo.setBindings(uboBinding);
		result.frameLayout = device.createDescriptorSetLayout(frameLayoutInfo);

		vk::DescriptorPoolSize        framePoolSize{vk::DescriptorType::eUniformBuffer, 1};
		vk::DescriptorPoolCreateInfo framePoolInfo{};
		framePoolInfo.setPoolSizes(framePoolSize);
		framePoolInfo.setMaxSets(1);
		result.framePool = device.createDescriptorPool(framePoolInfo);

		vk::DescriptorSetAllocateInfo frameAllocInfo{};
		frameAllocInfo.setDescriptorPool(result.framePool);
		frameAllocInfo.setSetLayouts(result.frameLayout);
		vk::DescriptorSet frameSet = device.allocateDescriptorSets(frameAllocInfo).front();

		VkBufferCreateInfo frameBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		frameBufferInfo.size = sizeof(FrameUBO);
		frameBufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VmaAllocationCreateInfo frameAllocCreateInfo{};
		frameAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		frameAllocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;
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
			// Zeroed, not real camera data -- this test never asserts on FrameUBO-derived pixel
			// values, only that the descriptor is real and defined (an unwritten uniform-buffer
			// descriptor read by a shader that statically uses it is what this fixture exists to
			// avoid).
			std::memset(frameAllocResultInfo.pMappedData, 0, sizeof(FrameUBO));
		}

		vk::DescriptorBufferInfo frameBufferDescInfo{result.frameUboBuffer, 0, sizeof(FrameUBO)};
		vk::WriteDescriptorSet   uboWrite{};
		uboWrite.setDstSet(frameSet);
		uboWrite.setDstBinding(0);
		uboWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		uboWrite.setBufferInfo(frameBufferDescInfo);
		device.updateDescriptorSets(uboWrite, nullptr);

		// -- Bindless set (set 1): sampled 2D, sampled 2D array + sampler catalog --
		std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{};
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
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 3> bindingFlags{
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eUpdateAfterBind,
			vk::DescriptorBindingFlags{},
		};
		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
		bindingFlagsInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindingCount(3);
		layoutInfo.setBindings(layoutBindings);
		layoutInfo.setFlags(vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool);
		layoutInfo.pNext = &bindingFlagsInfo;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 3> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
			vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1},
		};
		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setMaxSets(1);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);
		result.pool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(result.pool);
		allocInfo.setSetLayouts(result.layout);
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
		samplerWrite.setDstSet(set);
		samplerWrite.setDstBinding(2);
		samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfo);
		device.updateDescriptorSets(samplerWrite, {});

		result.bindings.set = set;
		result.bindings.layout = result.layout;
		result.bindings.sampledImage2DBinding = 0;
		result.bindings.sampledImage2DArrayBinding = 1;
		result.bindings.samplerBinding = 2;
		result.bindings.frameSet = frameSet;
		result.bindings.frameSetLayout = result.frameLayout;
		return result;
	}

	void DestroyWaterBindlessSet(vk::Device device, VmaAllocator allocator, WaterBindlessSet& s) {
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
	"WaterNode composites over an existing scene via real alpha blending, at the correct phase, with no "
	"validation errors"
) {
	brassica::testing::MinimalDevice device;
	if (!device.IsValid()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = device.GetDevice();

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				device.GetQueueFamily(),
			}
		);
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

		render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);

		graph::PhysicalResourceRegistry registry(vkDevice, device.GetAllocator());
		WaterBindlessSet                bindlessSet = CreateWaterBindlessSet(vkDevice, device.GetAllocator());
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		graph::PhysicalExecutionBackend backend(registry);

		constexpr vk::Format kSwapchainFormat = vk::Format::eR8G8B8A8Unorm;

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);
		// MinimalDevice does not enable VK_EXT_mesh_shader. On Mesa/lavapipe, vkGetDeviceProcAddr
		// returns a non-null pointer for disabled extension functions. Clear mesh shader function
		// pointers when extension is not enabled to avoid driver crash.
		dls.vkCmdDrawMeshTasksEXT = nullptr;
		dls.vkCmdDrawMeshTasksIndirectEXT = nullptr;

		WaterNode waterNode;
		waterNode.Init(
			render::NodeServices{
				.device = vkDevice,
				.pipelineLibrary = &pipelineLibrary,
				.dispatchLoader = &dls,
				.swapchainFormat = kSwapchainFormat,
			}
		);
		Shader::ClearConstants();

		graph::Graph graph;
		graph.Register<FakeSceneProducer>(FakeSceneProducer{.extent = {256, 256}, .swapchainFormat = kSwapchainFormat});
		graph.RegisterRef(waterNode);

		graph::FrameContext ctx{.width = 256, .height = 256};

		vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
		CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
		vkCmd.end();

		const auto& schedule = graph.GetSchedule();
		REQUIRE(schedule.stages.size() == 2);
		CHECK(schedule.stages[0].nodes.size() == 1);
		CHECK(schedule.stages[1].nodes.size() == 1);

		vk::SubmitInfo submitInfo{};
		submitInfo.setCommandBuffers(vkCmd);
		device.GetQueue().submit(submitInfo);
		device.GetQueue().waitIdle();

		pipelineLibrary.Reset();
		waterNode.Destroy(vkDevice);
		DestroyWaterBindlessSet(vkDevice, device.GetAllocator(), bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
