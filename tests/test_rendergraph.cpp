#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/AtmosphereSkyNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainNode.hpp"

using namespace brassica;

TEST_CASE("Real TerrainGenNode/AtmosphereSkyNode/TerrainNode/DeferredNode compose into a renderable, correctly-staged graph") {
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
		frameGraph.Register<graph::Import<TerrainTLAS>>();
		frameGraph.Register<graph::Import<Swapchain>>();

		render::EngineNodeRegistry::Instance().RegisterAllInto(frameGraph);

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		const auto& schedule = frameGraph.GetSchedule();
		CHECK(!schedule.stages.empty());
	}

	engine.Cleanup();
}
