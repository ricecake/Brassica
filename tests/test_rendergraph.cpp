#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/GradientNode.hpp"
#include "passes/TerrainNode.hpp"

using namespace brassica;

TEST_CASE("Real GradientNode/TerrainNode/DeferredNode compose into a renderable, correctly-staged graph") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	{
		graph::Graph frameGraph;
		frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
		frameGraph.Register<graph::Import<TerrainMinMaxTexture>>();
		frameGraph.Register<graph::Import<TerrainBiomeTexture>>();
		frameGraph.Register<graph::Import<TerrainTileVisibilityTexture>>();
		frameGraph.Register<graph::Import<TerrainIndirectionMapTexture>>();

		frameGraph.Register<GradientNode>(GradientNode{});
		frameGraph.Register<TerrainNode>(TerrainNode{});
		frameGraph.Register<DeferredNode>(DeferredNode{
			.swapchainFormat = vk::Format::eB8G8R8A8Unorm,
		});

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		const auto& schedule = frameGraph.GetSchedule();
		REQUIRE(schedule.stages.size() == 3);
		CHECK(schedule.stages[0].nodes.size() == 6);
		CHECK(schedule.stages[1].nodes.size() == 1);
		CHECK(schedule.stages[2].nodes.size() == 1);
	}

	engine.Cleanup();
}
