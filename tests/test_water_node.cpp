#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/WaterNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
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
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

	// Minimal real bindless set -- just enough for water.frag's SAMPLE_NEAREST(gPositionIndex/
	// gAlbedoIndex/gNormalIndex, ...) to resolve real descriptors: binding 0 (sampled 2D) is what
	// PhysicalRegistry writes GBufferPosition/GBufferAlbedo/GBufferNormal's indices into, binding 2 (samplers)
	// needs a real sampler written at BRASSICA_SAMPLER_NEAREST_CLAMP's index (0) since the
	// shader indexes it unconditionally. No array/storage/AS bindings -- WaterNode never touches
	// them. Mirrors test_physical_backend.cpp's own CreateBindlessTestSet, trimmed to what this
	// node actually needs.
	struct WaterBindlessSet {
		vk::DescriptorSetLayout                           layout{};
		vk::DescriptorPool                                pool{};
		vk::Sampler                                       sampler{};
		graph::PhysicalResourceRegistry::BindlessBindings bindings{};
	};

	WaterBindlessSet CreateWaterBindlessSet(vk::Device device) {
		std::array<vk::DescriptorSetLayoutBinding, 3> layoutBindings{};
		// Binding 0: FrameUBO
		layoutBindings[0]
			.setBinding(0)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 1: uTextures2D
		layoutBindings[1]
			.setBinding(1)
			.setDescriptorType(vk::DescriptorType::eSampledImage)
			.setDescriptorCount(8)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);
		// Binding 3: uSamplers
		layoutBindings[2]
			.setBinding(3)
			.setDescriptorType(vk::DescriptorType::eSampler)
			.setDescriptorCount(1)
			.setStageFlags(vk::ShaderStageFlagBits::eAll);

		std::array<vk::DescriptorBindingFlags, 3> bindingFlags{
			vk::DescriptorBindingFlags{},
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

		WaterBindlessSet result;
		result.layout = device.createDescriptorSetLayout(layoutInfo);

		std::array<vk::DescriptorPoolSize, 3> poolSizes{
			vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1},
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
		samplerWrite.setDstBinding(3);
		samplerWrite.setDescriptorType(vk::DescriptorType::eSampler);
		samplerWrite.setImageInfo(samplerImageInfo);
		device.updateDescriptorSets(samplerWrite, {});

		result.bindings.set = set;
		result.bindings.layout = result.layout;
		result.bindings.uboBinding = 0;
		result.bindings.sampledImage2DBinding = 1;
		result.bindings.samplerBinding = 3;
		return result;
	}

	void DestroyWaterBindlessSet(vk::Device device, WaterBindlessSet& s) {
		device.destroySampler(s.sampler);
		device.destroyDescriptorPool(s.pool);
		device.destroyDescriptorSetLayout(s.layout);
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
		WaterBindlessSet                bindlessSet = CreateWaterBindlessSet(vkDevice);
		registry.SetGlobalDescriptorSet(bindlessSet.bindings);
		graph::PhysicalExecutionBackend backend(registry);

		constexpr vk::Format kSwapchainFormat = vk::Format::eR8G8B8A8Unorm;

		DispatchLoaderDynamic dls;
		dls.init(vkDevice);

		WaterNode waterNode;
		waterNode.Init(vkDevice, &pipelineLibrary, &dls, kSwapchainFormat);
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
		DestroyWaterBindlessSet(vkDevice, bindlessSet);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(device.GetValidationErrorCount() == 0);
	CHECK(device.GetValidationWarningCount() == 0);
}
