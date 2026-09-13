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

// Real GradientNode/TerrainNode/DeferredNode registered into a real graph::Graph -- genuine
// integration coverage the old fg::-based version of this test never had (it exercised
// hand-written MockGradientPass/MockGBufferPass/MockDeferredPass lambdas, never the real pass
// classes). Gated on a real headless device, same pattern as tests/test_headless.cpp/
// test_physical_backend.cpp -- none of these three nodes need a real device to construct
// anymore (no more Pass classes building a real Vulkan pipeline in their constructor), but
// engine.Init() itself still does elsewhere in the engine, so this stays gated the same way.
//
// Only Setup()+Compile() are exercised here, matching the old test's own scope (it never called
// fg.execute() either) -- this proves the real production nodes compose into a renderable,
// correctly-scheduled graph; PhysicalExecutionBackend's actual Provision/barrier/render path has
// its own dedicated coverage in tests/test_physical_backend.cpp.
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
		// TerrainClipmapTexture has no producer node -- it's registered once, directly into the
		// registry, at Engine::Init (unexercised here, this test never provisions/executes) --
		// so a plain Import declares it for Compile()'s validation, matching Engine.cpp's real
		// per-frame graph exactly.
		frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
		// pipelineLibrary/shader/terrainAS pointers stay null on all three nodes -- this test's
		// scope is Setup()+Compile() only (see the header comment above), and Execute (the only
		// place those fields are read) is never called here.
		frameGraph.Register<GradientNode>(GradientNode{});
		frameGraph.Register<TerrainNode>(TerrainNode{});
		frameGraph.Register<DeferredNode>(DeferredNode{
			.swapchainFormat = vk::Format::eB8G8R8A8Unorm,
		});

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		// Real cross-node edges, not just "it compiled": Gradient and the clipmap Import share
		// no dependency with anything, so they land in the first stage (provably independent);
		// Terrain now declares Read<TerrainClipmapTexture> (a real, declared dependency rather
		// than a raw view/sampler smuggled in with no graph edge), so it must land strictly
		// after the Import; Deferred depends on Terrain's outputs too, so it lands later still.
		const auto& schedule = frameGraph.GetSchedule();
		REQUIRE(schedule.stages.size() == 3);
		CHECK(schedule.stages[0].nodes.size() == 2);
		CHECK(schedule.stages[1].nodes.size() == 1);
		CHECK(schedule.stages[2].nodes.size() == 1);
	}

	engine.Cleanup();
}
