#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereSkyNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainNode.hpp"

using namespace brassica;

// Real TerrainGenNode/GradientNode/TerrainNode/DeferredNode registered into a real graph::Graph -- genuine
// integration coverage the old fg::-based version of this test never had (it exercised
// hand-written MockGradientPass/MockGBufferPass/MockDeferredPass lambdas, never the real pass
// classes). Gated on a real headless device, same pattern as tests/test_headless.cpp/
// test_physical_backend.cpp -- none of these nodes need a real device to construct
// anymore (no more Pass classes building a real Vulkan pipeline in their constructor), but
// engine.Init() itself still does elsewhere in the engine, so this stays gated the same way.
//
// Only Setup()+Compile() are exercised here, matching the old test's own scope (it never called
// fg.execute() either) -- this proves the real production nodes compose into a renderable,
// correctly-scheduled graph; PhysicalExecutionBackend's actual Provision/barrier/render path has
// its own dedicated coverage in tests/test_physical_backend.cpp.
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

		// pipelineLibrary/shader/terrainAS pointers stay null on all nodes -- this test's
		// scope is Setup()+Compile() only (see the header comment above), and Execute (the only
		// place those fields are read) is never called here.
		frameGraph.Register<TerrainGenNode>(TerrainGenNode{});
		frameGraph.Register<AtmosphereSkyNode>(AtmosphereSkyNode{});
		frameGraph.Register<TerrainNode>(TerrainNode{});
		frameGraph.Register<DeferredNode>(DeferredNode{
			.swapchainFormat = vk::Format::eB8G8R8A8Unorm,
		});

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		const auto& schedule = frameGraph.GetSchedule();
		CHECK(!schedule.stages.empty());
	}

	engine.Cleanup();
}
