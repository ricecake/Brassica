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
	// FoliageBiomeTableTexture's real producer is a PredefinedTextureNode instance owned by
	// FoliageManager (FoliageManager::RegisterTableNode, called alongside RegisterAllInto in
	// Engine.cpp's real graph builds) -- not part of EngineNodeRegistry's auto-registered set,
	// so this test has no FoliageManager to register it the real way. Same category as the
	// other Import<>s above: a key whose producer lives outside the auto-registered set.
	frameGraph.Register<graph::Import<FoliageBiomeTableTexture>>();
	// Same category again: GltfVertexBuffer/GltfMeshletVertexBuffer/GltfMeshletTriangleBuffer/
	// GltfMeshletBuffer/ModelClipPaletteBuffer/ModelClipTableBuffer/ModelDescBuffer/
	// ModelSkeletonBuffer's real producers are ModelRegistry's PredefinedBufferNode members
	// (ModelRegistry::RegisterBufferNodes -- Stage 2 moved these off GltfModelManager, which is
	// now just one of ModelRegistry's producers; ModelSkeletonBuffer is Stage 4's), and
	// ModelInstanceWriteBuffer's is ModelInstanceManager's, and MaterialBuffer's is
	// MaterialManager's (MaterialManager::RegisterBufferNode) -- all called alongside
	// RegisterAllInto in Engine.cpp's real graph builds, none part of EngineNodeRegistry's
	// auto-registered set. ModelRenderNode/ModelInstanceScatterNode/ModelCullNode/
	// ModelLiveEvalNode (all auto-registered) are the real consumers once ForceAllActive() below
	// stops letting their own Setup()-decided inactivity (no ModelRegistry/ModelInstanceManager
	// service in this test) hide the gap. ModelInstanceBuffer and Stage 3/4's own
	// ModelVisibleListBuffer/ModelIndirectArgsBuffer/ModelLiveListBuffer/
	// ModelLiveIndirectArgsBuffer/ModelLivePaletteBuffer are all catalog-converted (no Import<>
	// needed -- their real producers, ModelInstanceScatterNode/ModelCullResetNode/ModelCullNode/
	// ModelLiveEvalNode, are themselves auto-registered, so the catalog already resolves them).
	frameGraph.Register<graph::Import<GltfVertexBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletVertexBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletTriangleBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletBuffer>>();
	frameGraph.Register<graph::Import<ModelClipPaletteBuffer>>();
	frameGraph.Register<graph::Import<ModelClipTableBuffer>>();
	frameGraph.Register<graph::Import<ModelDescBuffer>>();
	frameGraph.Register<graph::Import<ModelSkeletonBuffer>>();
	frameGraph.Register<graph::Import<ModelInstanceWriteBuffer>>();
	frameGraph.Register<graph::Import<MaterialBuffer>>();

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
