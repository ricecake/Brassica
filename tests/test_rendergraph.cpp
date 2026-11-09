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
	// Re-captured after untangling the cloud-revamp/model-work and major-graph-rework branches
	// onto one tree (main-unified): the particle cluster's reactivation (S2, passes/particle/
	// resources.hpp's header comment -- ForceAllActive() overrides IParticleManager::
	// IsEnabled(), so all 7 particle nodes appear here even though they're runtime-gated in
	// production) landed alongside the independently-developed cloud/model nodes (CloudVolume*,
	// CloudShadow/Render/Composite/Temporal, FoliageBake/IndirectReset/ArchetypeRender,
	// TerrainShadowNode, GltfRenderNode, LightningAge/RenderNode) for the first time. Expected
	// diff from the pre-untangling fixture: every one of those additions, plus
	// AtmosphereCompositeNode's removal (unified into VolumetricRenderNode, predates this
	// session) and CloudRenderNode moving from the HdrColor writer chain to a compute pass
	// feeding CloudCompositeNode instead.
	const std::string kGoldenScheduleFixture =
		R"FIXTURE(STAGE 0:
  brassica::graph::Import<brassica::AutoExposureBuffer>
  brassica::graph::Import<brassica::ClusteredLighting>
  brassica::graph::Import<brassica::FoliageBiomeTableTexture>
  brassica::graph::Import<brassica::GltfJointMatrixBuffer>
  brassica::graph::Import<brassica::GltfMeshletBuffer>
  brassica::graph::Import<brassica::GltfMeshletTriangleBuffer>
  brassica::graph::Import<brassica::GltfMeshletVertexBuffer>
  brassica::graph::Import<brassica::GltfVertexBuffer>
  brassica::graph::Import<brassica::LightsBuffer>
  brassica::graph::Import<brassica::MaterialBuffer>
  brassica::graph::Import<brassica::Swapchain>
  brassica::graph::Import<brassica::TerrainTLAS>
  brassica::graph::Import<brassica::graph::History<brassica::HiZTexture> >
STAGE 1:
  brassica::ParticleTypeBufferNode
STAGE 2:
  brassica::CloudVolumeGenNode
  brassica::EntityPrepareNode
  brassica::FoliageBakeNode
  brassica::FoliageIndirectResetNode
  brassica::ParticleResetNode
  brassica::TerrainGenNode
  brassica::TransmittanceLUTNode
STAGE 3:
  brassica::CloudVolumeDownsampleNode
  brassica::MultiScatteringLUTNode
  brassica::TerrainMinMaxDownsampleNode
  brassica::TerrainReadbackNode
STAGE 4:
  brassica::ParticleLivenessNode
  brassica::SkyViewLUTNode
  brassica::TerrainShadowNode
STAGE 5:
  brassica::ParticleGridBuildNode
STAGE 6:
  brassica::ParticleBehaviorNode
STAGE 7:
  brassica::TerrainBiomeNode
STAGE 8:
  brassica::CloudShadowGenNode
  brassica::LightningAgeNode
  brassica::SkyBackgroundNode
  brassica::TerrainNode
STAGE 9:
  brassica::ClusterLightAssignmentNode
  brassica::EntityNode
  brassica::VolumetricFroxelGenNode
STAGE 10:
  brassica::FoliageNode
STAGE 11:
  brassica::FoliageArchetypeRenderNode
STAGE 12:
  brassica::GltfRenderNode
STAGE 13:
  brassica::CloudRenderNode
  brassica::GTAONode
STAGE 14:
  brassica::CloudTemporalNode
  brassica::DeferredNode
STAGE 15:
  brassica::UnderwaterParticleRenderNode
STAGE 16:
  brassica::LightningRenderNode
STAGE 17:
  brassica::WaterNode
STAGE 18:
  brassica::AboveWaterParticleRenderNode
STAGE 19:
  brassica::CirrusNode
  brassica::HiZDownsampleNode
STAGE 20:
  brassica::CloudCompositeNode
STAGE 21:
  brassica::VolumetricRenderNode
STAGE 22:
  brassica::TonemapComputeNode
STAGE 23:
  brassica::TonemapNode
STAGE 24:
  brassica::ImGuiNode
MULTI-WRITER KEYS:
  brassica::AboveWaterParticleIndirectBuffer: brassica::ParticleLivenessNode, brassica::ParticleResetNode
  brassica::AutoExposureBuffer: brassica::graph::Import<brassica::AutoExposureBuffer>, brassica::TonemapComputeNode
  brassica::CloudDetailVolume: brassica::CloudVolumeDownsampleNode, brassica::CloudVolumeGenNode
  brassica::ClusteredLighting: brassica::graph::Import<brassica::ClusteredLighting>, brassica::ClusterLightAssignmentNode
  brassica::FoliageIndirectArgsBuffer: brassica::FoliageIndirectResetNode, brassica::FoliageNode
  brassica::GBufferAlbedo: brassica::EntityNode, brassica::FoliageArchetypeRenderNode, brassica::FoliageNode, brassica::GltfRenderNode, brassica::TerrainNode
  brassica::GBufferDepth: brassica::EntityNode, brassica::FoliageArchetypeRenderNode, brassica::FoliageNode, brassica::GltfRenderNode, brassica::AboveWaterParticleRenderNode, brassica::UnderwaterParticleRenderNode, brassica::TerrainNode
  brassica::GBufferMaterial: brassica::EntityNode, brassica::GltfRenderNode, brassica::TerrainNode
  brassica::GBufferNormal: brassica::EntityNode, brassica::FoliageArchetypeRenderNode, brassica::FoliageNode, brassica::GltfRenderNode, brassica::TerrainNode
  brassica::GBufferPosition: brassica::EntityNode, brassica::FoliageArchetypeRenderNode, brassica::FoliageNode, brassica::GltfRenderNode, brassica::TerrainNode
  brassica::HdrColor: brassica::CirrusNode, brassica::CloudCompositeNode, brassica::DeferredNode, brassica::LightningRenderNode, brassica::AboveWaterParticleRenderNode, brassica::UnderwaterParticleRenderNode, brassica::VolumetricRenderNode, brassica::WaterNode
  brassica::LightningStrikePool: brassica::TerrainBiomeNode, brassica::LightningAgeNode
  brassica::LightsBuffer: brassica::graph::Import<brassica::LightsBuffer>, brassica::LightningAgeNode, brassica::ParticleBehaviorNode
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
	frameGraph.Register<graph::Import<LightsBuffer>>();
	frameGraph.Register<graph::Import<ClusteredLighting>>();
	frameGraph.Register<graph::Import<graph::History<HiZTexture>>>();
	// FoliageBiomeTableTexture's real producer is a PredefinedTextureNode instance owned by
	// FoliageManager (FoliageManager::RegisterTableNode, called alongside RegisterAllInto in
	// Engine.cpp's real graph builds) -- not part of EngineNodeRegistry's auto-registered set,
	// so this test has no FoliageManager to register it the real way. Same category as the
	// other Import<>s above: a key whose producer lives outside the auto-registered set.
	frameGraph.Register<graph::Import<FoliageBiomeTableTexture>>();
	// Same category again: GltfVertexBuffer/GltfMeshletVertexBuffer/GltfMeshletTriangleBuffer/
	// GltfMeshletBuffer/GltfJointMatrixBuffer's real producers are GltfModelManager's
	// PredefinedBufferNode members (GltfModelManager::RegisterBufferNodes), and MaterialBuffer's
	// is MaterialManager's (MaterialManager::RegisterBufferNode) -- both called alongside
	// RegisterAllInto in Engine.cpp's real graph builds, neither part of EngineNodeRegistry's
	// auto-registered set. GltfRenderNode (auto-registered) is the one real consumer of all six
	// once ForceAllActive() below stops letting its own Setup()-decided inactivity (no
	// GltfModelManager service in this test) hide the gap.
	frameGraph.Register<graph::Import<GltfVertexBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletVertexBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletTriangleBuffer>>();
	frameGraph.Register<graph::Import<GltfMeshletBuffer>>();
	frameGraph.Register<graph::Import<GltfJointMatrixBuffer>>();
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
