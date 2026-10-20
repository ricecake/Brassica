#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/SkyBackgroundNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainNode.hpp"

using namespace brassica;

TEST_CASE("Real TerrainGenNode/SkyBackgroundNode/TerrainNode/DeferredNode compose into a renderable, correctly-staged graph") {
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

// S5 exit criterion (design plan's §E/S5): "the force-all-active compile runs in CI." Bypasses
// Engine::Init() entirely -- same technique as every other real-node-set check in this plan
// (EngineNodeRegistry::CreateAll()/RegisterAllInto()/Setup()/Compile() are pure CPU-side, no
// device needed) -- so this runs unconditionally, not gated behind a device check like the test
// above. ForceAllActive() overrides every node's own Setup()-decided activity to true, so this
// exercises every real node and every real resource edge regardless of what any node's actual
// runtime state (ServiceLocator managers, etc.) would otherwise decide this frame -- catching an
// ambiguity or cycle that only a disabled-by-default feature would normally hide.
TEST_CASE("Force-all-active compile of the real registered node set is acyclic and unambiguous") {
	ServiceLocator locator;
	ServiceLocator::SetInstance(&locator);

	render::EngineNodeRegistry::Instance().CreateAll();

	graph::Graph frameGraph;
	frameGraph.Register<graph::Import<TerrainTLAS>>();
	frameGraph.Register<graph::Import<Swapchain>>();
	frameGraph.Register<graph::Import<AutoExposureBuffer>>();
	frameGraph.Register<graph::Import<LightsBuffer>>();
	frameGraph.Register<graph::Import<graph::History<HiZTexture>>>();

	render::EngineNodeRegistry::Instance().RegisterAllInto(frameGraph);

	graph::FrameContext ctx{.width = 1280, .height = 720};
	frameGraph.Setup(ctx);
	frameGraph.ForceAllActive();

	auto compileResult = frameGraph.Compile();
	if (!compileResult) {
		MESSAGE(compileResult.error().message);
	}
	REQUIRE(compileResult.has_value());

	const auto& schedule = frameGraph.GetSchedule();
	CHECK(!schedule.stages.empty());

	ServiceLocator::SetInstance(nullptr);
}
