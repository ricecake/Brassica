#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/CirrusNode.hpp"
#include "passes/SkyBackgroundNode.hpp"
#include "passes/TerrainBiomeNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

using namespace brassica;

namespace {

	struct FakeSceneProducer {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferDepth>,
			graph::Create<HdrColor>>;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
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
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

	struct FakeWeatherProducer {
		using Resources = graph::Declares<
			graph::Create<TerrainWeatherBiomeTexture>>;

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherBiomeTexture>(),
					.access = graph::AccessKind::Write,
					.desc = WeatherBiomeImageDesc(256),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext&) {}
	};

} // namespace

TEST_CASE("CirrusNode renders through PhysicalExecutionBackend with no validation errors, across two frames") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = engine.GetDevice();
	vk::Queue  queue = vkDevice.getQueue(0, 0);

	{
		vk::CommandPool pool = vkDevice.createCommandPool(
			vk::CommandPoolCreateInfo{
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				0,
			}
		);
		vk::CommandBuffer vkCmd = vkDevice
									  .allocateCommandBuffers(
										  vk::CommandBufferAllocateInfo{pool, vk::CommandBufferLevel::ePrimary, 1}
									  )
									  .front();

		render::PipelineLibrary pipelineLibrary(vkDevice, engine.GetPipelineCache());

		TransmittanceLUTNode transNode;
		transNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		MultiScatteringLUTNode multiNode;
		multiNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		SkyViewLUTNode skyViewNode;
		skyViewNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		SkyBackgroundNode backgroundNode;
		backgroundNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});
		CirrusNode cirrusNode;
		cirrusNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});

		graph::PhysicalResourceRegistry& registry = engine.GetPhysicalRegistry();
		graph::PhysicalExecutionBackend  backend(registry);

		for (std::uint64_t frameIndex = 0; frameIndex < 2; ++frameIndex) {
			graph::Graph graph;
			graph.Register<FakeSceneProducer>(FakeSceneProducer{});
			graph.Register<FakeWeatherProducer>(FakeWeatherProducer{});
			graph.RegisterRef(transNode);
			graph.RegisterRef(multiNode);
			graph.RegisterRef(skyViewNode);
			graph.RegisterRef(backgroundNode);
			graph.RegisterRef(cirrusNode);

			graph::FrameContext ctx{.width = 256, .height = 256, .frameIndex = frameIndex};

			vkCmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			graph::CommandBuffer cmd{static_cast<void*>(static_cast<VkCommandBuffer>(vkCmd))};
			CHECK_NOTHROW(backend.Execute(graph, ctx, cmd, false));
			vkCmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(vkCmd);
			queue.submit(submitInfo);
			queue.waitIdle();
		}

		CHECK(registry.GetTexture<HdrColor>() != nullptr);

		pipelineLibrary.Reset();
		cirrusNode.Destroy(vkDevice);
		backgroundNode.Destroy(vkDevice);
		skyViewNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	engine.Cleanup();
}
