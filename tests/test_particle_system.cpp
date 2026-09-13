#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "MinimalDevice.hpp"
#include "passes/ParticleSystemNode.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

using namespace brassica;

namespace {

	struct DeferredShadingProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>,
			graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = graph::Phase::Default;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

} // namespace

TEST_CASE("Particle struct layouts match std430 16-byte alignment requirements") {
	CHECK(sizeof(Particle) == 64);
	CHECK(offsetof(Particle, position) == 0);
	CHECK(offsetof(Particle, velocity) == 16);
	CHECK(offsetof(Particle, misc) == 32);
	CHECK(offsetof(Particle, type) == 48);
	CHECK(offsetof(Particle, lifetime) == 52);
	CHECK(offsetof(Particle, maxLifetime) == 56);
	CHECK(offsetof(Particle, padding) == 60);

	CHECK(sizeof(ParticleType) == 32);
	CHECK(sizeof(ParticleIndirectCommand) == 12);
}

TEST_CASE("ParticleSystemNode executes in Phase::Late after DeferredNode in Phase::Default") {
	graph::Graph graph;
	graph.Register<DeferredShadingProducer>();
	graph.Register<graph::Import<ParticleTypeBuffer>>();

	ParticleSystemNode particleNode;
	graph.RegisterRef(particleNode);

	graph::FrameContext ctx{.width = 256, .height = 256};
	graph.Setup(ctx);

	auto result = graph.Compile();
	if (!result) {
		MESSAGE(result.error().message);
	}
	REQUIRE(result.has_value());

	const auto& schedule = graph.GetSchedule();
	REQUIRE(schedule.stages.size() >= 2);

	std::size_t deferredStage = 0;
	std::size_t particleStage = 0;

	const auto& nodes = graph.Nodes();
	for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
		for (std::size_t nodeIdx : schedule.stages[s].nodes) {
			if (nodes[nodeIdx].Descriptor().name.find("DeferredShadingProducer") != std::string_view::npos) {
				deferredStage = s;
			}
			if (nodes[nodeIdx].Descriptor().name.find("ParticleSystemNode") != std::string_view::npos) {
				particleStage = s;
			}
		}
	}

	CHECK(particleStage > deferredStage);
}

TEST_CASE("ParticleSystemNode shader and pass initialization validation") {
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

		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

		render::PipelineLibrary pipelineLibrary(vkDevice, nullptr);

		std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
		bindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		bindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		bindings[2].setBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
		bindings[3].setBinding(3).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		vk::DescriptorSetLayout particleSetLayout = vkDevice.createDescriptorSetLayout(layoutInfo);

		DispatchLoaderDynamic dls;
		dls.init(device.GetInstance(), vkDevice);
		dls.vkCmdDrawMeshTasksEXT = nullptr;
		dls.vkCmdDrawMeshTasksIndirectEXT = nullptr;

		ParticleResetNode resetNode;
		resetNode.Init(vkDevice, &pipelineLibrary, particleSetLayout, nullptr);

		ParticleLivenessNode livenessNode;
		livenessNode.Init(vkDevice, &pipelineLibrary, particleSetLayout, nullptr);

		ParticleBehaviorNode behaviorNode;
		behaviorNode.Init(vkDevice, &pipelineLibrary, particleSetLayout, nullptr);

		ParticleRenderNode renderNode;
		renderNode.Init(vkDevice, &pipelineLibrary, &dls, vk::Format::eR8G8B8A8Unorm, particleSetLayout, nullptr);

		resetNode.Destroy(vkDevice);
		livenessNode.Destroy(vkDevice);
		behaviorNode.Destroy(vkDevice);
		renderNode.Destroy(vkDevice);

		Shader::ClearConstants();

		vkDevice.destroyDescriptorSetLayout(particleSetLayout);
		vkDevice.destroyCommandPool(pool);
	}
}
