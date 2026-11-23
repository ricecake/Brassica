#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

// 13 -> 15: UnderwaterParticleRenderNode/AboveWaterParticleRenderNode
// 16 -> 17: Added CirrusNode for thin cirrus cloud weather indicator layer.
// 18 -> 20: Added CloudVolumeGenNode and CloudRenderNode.
// 20 -> 24: Replaced per-handler EntityNode with auto-registered EntityPrepareNode and EntityNode.
// 24 -> 25: Added HiZDownsampleNode.
// 25 -> 27: Added VolumetricFroxelGenNode and VolumetricRenderNode for froxel volumetric rendering.
// 27 -> 28: Added GTAONode.
// 28 -> 26: ParticleSystemNode sub-nodes dynamically registered, added CloudShadowGenNode.
// 26 -> 27: Added TerrainReadbackNode
// 27 -> 28: Added TerrainShadowNode.
// 28 -> 27: Unified AtmosphereCompositeNode into VolumetricRenderNode.
// 27 -> 32: Drift
// 33 -> 34: Added FoliageIndirectResetNode (splitting FoliageIndirectArgsBuffer's zero-clear out
// of FoliageNode::Execute() into its own Compute-domain node -- see FoliageNode.hpp's comment;
// the Graphics-domain original was issuing an illegal barrier+fillBuffer inside its own
// already-open dynamic-rendering scope, confirmed via real GPU-assisted validation).
// 34 -> 43: Particle cluster reactivated (S2) -- ParticleResetNode, ParticleLivenessNode,
// ParticleBehaviorNode, ParticleGridBuildNode, UnderwaterParticleRenderNode,
// AboveWaterParticleRenderNode, ParticleTypeBufferNode all uncommented, gated at runtime by
// IParticleManager::IsEnabled().
// 43 -> 45: Resynced with auto-registered nodes (including TonemapComputeNode / TonemapNode).
// 45 -> 46: Added GroundDebrisNode for camera proximity terrain debris mesh rendering.
// 46 -> 48: Added GpuLightsResetNode (CPU/GPU light buffer split -- zeroes GpuLightsBuffer each
// frame so atomic-claimed dynamic light slots, e.g. fireflies, start from a known state) and
// ShadingRateComputeNode (attachment-only variable rate shading -- derives this frame's per-tile
// shading rate from last frame's ShadingRateMap/HiZTexture). This was main's own lineage.
// Model-work branch merge (generalized-model-mahem, rebased onto the above on 2026-11-10):
// replaced GltfRenderNode with ModelRenderNode (GPU-driven model instances, baked clips) and
// added ModelInstanceScatterNode (applies ModelInstanceManager's sparse CPU writes to
// ModelInstanceBuffer); added ModelCullResetNode and ModelCullNode (Stage 3 -- compacted-
// visible-list cull pass + indirect task dispatch); added ModelLiveFinalizeNode and
// ModelLiveEvalNode (Stage 4 -- live wind/seed-bend eval); added DecorPopulationNode (Stage 5.2
// -- persistent, disocclusion-driven decor population). Removed FoliageBakeNode,
// FoliageArchetypeRenderNode, and FoliageIndirectResetNode (Stage 5.1/5.2 cleanup -- the GPU-
// baked tree/bush archetype path they served is fully superseded by real
// ManifoldBranchingTreeAsset/ManifoldRockAsset models + DecorPopulationNode's persistent
// population). Two independent deltas from two different baselines (main's own 46->48, and the
// model branch's own -4/+7 relative to ITS OWN pre-merge baseline) don't combine by simple
// arithmetic -- resynced to the real current count via a direct grep of every
// BRASSICA_REGISTER_NODE( call site across AllNodes.hpp's own include list, not arithmetic,
// same "Drift"/"resynced" precedent already used twice above in this exact file.
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 49);
}
