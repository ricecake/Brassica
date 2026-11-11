#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <map>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/atmosphere/nodes/AtmosphereLUTNode.hpp"
#include "passes/atmosphere/nodes/SkyBackgroundNode.hpp"
#include "passes/scene/nodes/DeferredNode.hpp"
#include "passes/terrain/nodes/TerrainGenNode.hpp"
#include "passes/terrain/nodes/TerrainNode.hpp"

using namespace brassica;

namespace {

	// S0 golden fixture (design plan's restructure, "zippy-seeking-hanrahan"): a layout-only move
	// (S1) or a pure resource-type conversion (S2-S4, S6) must reproduce this byte-identical --
	// any diff means something that was supposed to be a no-op logic change actually moved a node
	// between stages or changed which node wins a multi-writer key. S5 (managers own nodes) is
	// expected to change the MULTI-WRITER KEYS section's import list once ClusteredLighting/etc.
	// become real imports and the two drifted Engine.cpp graph-build sites unify -- see the plan's
	// S0/S5 sections for what's expected to change there.
	//
	// Stage membership is sorted alphabetically within a stage (node order within a stage is
	// unconstrained -- two nodes share a stage specifically because neither depends on the
	// other -- so an unsorted dump would make this fixture flaky against nothing more than
	// std::vector's own incidental ordering). Multi-writer keys are listed in node-registration
	// order, which is what PhysicalRegistry::Provision's schedule-last-wins collapse (lessons
	// 7/10/11/12a, project-brassica-graph-phase-lessons) actually iterates.
	std::string DumpScheduleFixture(const graph::Graph& g) {
		std::string      out;
		const auto       nodes = g.Nodes();
		const auto&      schedule = g.GetSchedule();

		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			std::vector<std::string> names;
			for (std::size_t idx : schedule.stages[s].nodes) {
				names.emplace_back(nodes[idx].Descriptor().name);
			}
			std::sort(names.begin(), names.end());
			out += "STAGE " + std::to_string(s) + ":\n";
			for (const auto& n : names) {
				out += "  " + n + "\n";
			}
		}

		std::map<std::string, std::vector<std::string>> writersByKeyName;
		for (const auto& node : nodes) {
			const auto desc = node.Descriptor();
			for (const graph::ResourceId key : desc.produces) {
				writersByKeyName[std::string(key->name)].emplace_back(desc.name);
			}
		}
		out += "MULTI-WRITER KEYS:\n";
		for (const auto& [keyName, writerNames] : writersByKeyName) {
			if (writerNames.size() < 2) {
				continue;
			}
			out += "  " + keyName + ": ";
			for (std::size_t i = 0; i < writerNames.size(); ++i) {
				if (i) {
					out += ", ";
				}
				out += writerNames[i];
			}
			out += "\n";
		}
		return out;
	}

	// Captured from a real run of the test below (DumpScheduleFixture's output); see its comment
	// for what this is allowed and not allowed to change across. Re-capture via
	// `ctest -R test_rendergraph --output-on-failure` (or run the binary directly with
	// `--test-case="Force-all-active*" --success`) and copy the printed "actual dump" back in
	// here if a stage intentionally changes.
	// Re-captured for the CPU/GPU light buffer split and ShadingRateComputeNode (main's own
	// lineage), then again after rebasing the model-work branch (generalized-model-mahem) onto
	// main on 2026-11-10 -- GltfRenderNode replaced by ModelRenderNode, FoliageBakeNode/
	// FoliageIndirectResetNode/FoliageArchetypeRenderNode deleted (superseded by
	// DecorPopulationNode's persistent population of real Manifold-generated models), the full
	// Model*/ModelInstanceWriteBuffer Import<> set and ModelCullResetNode/ModelInstanceScatterNode/
	// DecorPopulationNode/ModelCullNode/ModelLiveFinalizeNode/ModelLiveEvalNode/ModelRenderNode
	// added. Recaptured for real from a live run after resolving the rebase conflicts (see git
	// history for this file around 2026-11-10).
	const std::string kGoldenScheduleFixture =
		R"FIXTURE(STAGE 0:
  brassica::graph::Import<brassica::AutoExposureBuffer>
  brassica::graph::Import<brassica::ClusteredLighting>
  brassica::graph::Import<brassica::CpuLightsBuffer>
  brassica::graph::Import<brassica::FoliageBiomeTableTexture>
  brassica::graph::Import<brassica::GltfMeshletBuffer>
  brassica::graph::Import<brassica::GltfMeshletTriangleBuffer>
  brassica::graph::Import<brassica::GltfMeshletVertexBuffer>
  brassica::graph::Import<brassica::GltfVertexBuffer>
  brassica::graph::Import<brassica::MaterialBuffer>
  brassica::graph::Import<brassica::ModelClipPaletteBuffer>
  brassica::graph::Import<brassica::ModelClipTableBuffer>
  brassica::graph::Import<brassica::ModelDescBuffer>
  brassica::graph::Import<brassica::ModelInstanceWriteBuffer>
  brassica::graph::Import<brassica::ModelSkeletonBuffer>
  brassica::graph::Import<brassica::Swapchain>
  brassica::graph::Import<brassica::TerrainTLAS>
  brassica::graph::Import<brassica::graph::History<brassica::HiZTexture>>
STAGE 1:
  brassica::ParticleTypeBufferNode
STAGE 2:
  brassica::CloudVolumeGenNode
  brassica::EntityPrepareNode
  brassica::GpuLightsResetNode
  brassica::ModelCullResetNode
  brassica::ModelInstanceScatterNode
  brassica::ParticleResetNode
  brassica::ShadingRateComputeNode
  brassica::TerrainBiomeNode
  brassica::TerrainGenNode
  brassica::TransmittanceLUTNode
STAGE 3:
  brassica::CloudVolumeDownsampleNode
  brassica::DecorPopulationNode
  brassica::MultiScatteringLUTNode
  brassica::TerrainMinMaxDownsampleNode
  brassica::TerrainReadbackNode
STAGE 4:
  brassica::CloudShadowGenNode
  brassica::ModelCullNode
  brassica::ParticleLivenessNode
  brassica::SkyViewLUTNode
  brassica::TerrainNode
  brassica::TerrainShadowNode
STAGE 5:
  brassica::EntityNode
  brassica::ModelLiveFinalizeNode
  brassica::ParticleGridBuildNode
  brassica::SkyBackgroundNode
STAGE 6:
  brassica::FoliageNode
  brassica::ModelLiveEvalNode
  brassica::ParticleBehaviorNode
STAGE 7:
  brassica::GroundDebrisNode
  brassica::LightningAgeNode
STAGE 8:
  brassica::ClusterLightAssignmentNode
  brassica::ModelRenderNode
  brassica::VolumetricFroxelGenNode
STAGE 9:
  brassica::CloudRenderNode
  brassica::GTAONode
STAGE 10:
  brassica::CloudTemporalNode
  brassica::DeferredNode
STAGE 11:
  brassica::UnderwaterParticleRenderNode
STAGE 12:
  brassica::LightningRenderNode
STAGE 13:
  brassica::WaterNode
STAGE 14:
  brassica::AboveWaterParticleRenderNode
STAGE 15:
  brassica::CirrusNode
  brassica::HiZDownsampleNode
STAGE 16:
  brassica::CloudCompositeNode
STAGE 17:
  brassica::VolumetricRenderNode
STAGE 18:
  brassica::TonemapComputeNode
STAGE 19:
  brassica::TonemapNode
STAGE 20:
  brassica::ImGuiNode
MULTI-WRITER KEYS:
  brassica::AboveWaterParticleIndirectBuffer: brassica::ParticleLivenessNode, brassica::ParticleResetNode
  brassica::AutoExposureBuffer: brassica::graph::Import<brassica::AutoExposureBuffer>, brassica::TonemapComputeNode
  brassica::CloudDetailVolume: brassica::CloudVolumeDownsampleNode, brassica::CloudVolumeGenNode
  brassica::ClusteredLighting: brassica::graph::Import<brassica::ClusteredLighting>, brassica::ClusterLightAssignmentNode
  brassica::GBufferAlbedo: brassica::EntityNode, brassica::FoliageNode, brassica::ModelRenderNode, brassica::TerrainNode, brassica::GroundDebrisNode
  brassica::GBufferDepth: brassica::EntityNode, brassica::FoliageNode, brassica::ModelRenderNode, brassica::AboveWaterParticleRenderNode, brassica::UnderwaterParticleRenderNode, brassica::TerrainNode, brassica::GroundDebrisNode
  brassica::GBufferMaterial: brassica::EntityNode, brassica::ModelRenderNode, brassica::TerrainNode, brassica::GroundDebrisNode
  brassica::GBufferNormal: brassica::EntityNode, brassica::FoliageNode, brassica::ModelRenderNode, brassica::TerrainNode, brassica::GroundDebrisNode
  brassica::GBufferPosition: brassica::EntityNode, brassica::FoliageNode, brassica::ModelRenderNode, brassica::TerrainNode, brassica::GroundDebrisNode
  brassica::GpuLightsBuffer: brassica::GpuLightsResetNode, brassica::LightningAgeNode, brassica::ParticleBehaviorNode
  brassica::HdrColor: brassica::CirrusNode, brassica::CloudCompositeNode, brassica::DeferredNode, brassica::LightningRenderNode, brassica::AboveWaterParticleRenderNode, brassica::UnderwaterParticleRenderNode, brassica::VolumetricRenderNode, brassica::WaterNode
  brassica::LightningStrikePool: brassica::TerrainBiomeNode, brassica::LightningAgeNode
  brassica::ModelIndirectArgsBuffer: brassica::ModelCullResetNode, brassica::ModelCullNode
  brassica::ModelInstanceBuffer: brassica::DecorPopulationNode, brassica::ModelInstanceScatterNode
  brassica::ModelLiveIndirectArgsBuffer: brassica::ModelCullResetNode, brassica::ModelCullNode, brassica::ModelLiveFinalizeNode
  brassica::ParticleBuffer: brassica::ParticleBehaviorNode, brassica::ParticleLivenessNode
  brassica::ParticleGridHeadsBuffer: brassica::ParticleGridBuildNode, brassica::ParticleResetNode
  brassica::Swapchain: brassica::graph::Import<brassica::Swapchain>, brassica::ImGuiNode, brassica::TonemapNode
  brassica::TerrainMinMaxTexture: brassica::TerrainGenNode, brassica::TerrainMinMaxDownsampleNode
  brassica::TerrainTLAS: brassica::graph::Import<brassica::TerrainTLAS>, brassica::TerrainGenNode
  brassica::UnderwaterParticleIndirectBuffer: brassica::ParticleLivenessNode, brassica::ParticleResetNode
)FIXTURE";

} // namespace

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
	// CpuLightsBuffer's real producer is LightManager's own PredefinedBufferNode
	// (LightManager::RegisterBufferNode) -- same "manager-owned, not part of
	// EngineNodeRegistry's auto-registered set" category as MaterialBuffer/the Gltf buffers
	// below. GpuLightsBuffer needs no such Import<>: its real producer, GpuLightsResetNode, is
	// auto-registered (BRASSICA_REGISTER_NODE) and appears in the schedule for real.
	frameGraph.Register<graph::Import<CpuLightsBuffer>>();
	frameGraph.Register<graph::Import<ClusteredLighting>>();
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

	// Golden schedule fixture -- see DumpScheduleFixture's comment above. MESSAGE prints the
	// actual dump on failure so a deliberate (explained) change can be re-captured by hand.
	const std::string actualFixture = DumpScheduleFixture(frameGraph);
	if (actualFixture != kGoldenScheduleFixture) {
		MESSAGE("Golden schedule fixture mismatch -- actual dump:\n" << actualFixture);
	}
	CHECK(actualFixture == kGoldenScheduleFixture);

	ServiceLocator::SetInstance(nullptr);
}
