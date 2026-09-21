#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/AtmosphereCompositeNode.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

using namespace brassica;

TEST_CASE("AtmosphereCompositeNode renders through PhysicalExecutionBackend with no validation errors, across two frames") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device vkDevice = engine.GetDevice();
	vk::Queue queue = vkDevice.getQueue(0, 0);

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
		AtmosphereCompositeNode compositeNode;
		compositeNode.Init(render::NodeServices{.device = vkDevice, .pipelineLibrary = &pipelineLibrary});

		graph::PhysicalResourceRegistry& registry = engine.GetPhysicalRegistry();
		graph::PhysicalExecutionBackend backend(registry);

		for (std::uint64_t frameIndex = 0; frameIndex < 2; ++frameIndex) {
			graph::Graph graph;
			graph.RegisterRef(transNode);
			graph.RegisterRef(multiNode);
			graph.RegisterRef(skyViewNode);
			graph.RegisterRef(compositeNode);

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

		CHECK(registry.GetTexture<AtmosphereRadiance>() != nullptr);

		pipelineLibrary.Reset();
		compositeNode.Destroy(vkDevice);
		skyViewNode.Destroy(vkDevice);
		multiNode.Destroy(vkDevice);
		transNode.Destroy(vkDevice);
		vkDevice.destroyCommandPool(pool);
	}

	CHECK(engine.GetValidationErrorCount() == 0);
	CHECK(engine.GetValidationWarningCount() == 0);

	engine.Cleanup();
}
