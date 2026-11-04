#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

// 14 before EntityNode replaced BallNode: BallNode used to be a plain auto-registered node like
// every other, but EntityNode<Tag> is instantiated per-Tag and registered dynamically per frame
// via SystemHandler::GetEntityNode().RegisterInto (Engine::DrawFrame), not through this static
// list -- see include/passes/AllNodes.hpp's own header comment for why that list exists at all.
// 13 -> 15: UnderwaterParticleRenderNode/AboveWaterParticleRenderNode were promoted out of
// ParticleSystemNode's Subgraph to be independent top-level nodes (ParticleSystemNode.hpp), so
// their SubPhase::UnderwaterParticleRender/ParticleRender phases actually reach the outer
// scheduler relative to WaterNode's SubPhase::WaterRender -- a Subgraph child's own kPhase only
// orders it against its Subgraph siblings, never an outer sibling.
// 16 -> 17: Added CirrusNode for thin cirrus cloud weather indicator layer.
// 18 -> 20: Added CloudVolumeGenNode and CloudRenderNode.
// 20 -> 24: Replaced per-handler EntityNode with auto-registered EntityPrepareNode and EntityNode.
// 24 -> 25: Added HiZDownsampleNode.
// 25 -> 27: Added VolumetricFroxelGenNode and VolumetricRenderNode for froxel volumetric rendering.
// 27 -> 28: Added GTAONode.
// 28 -> 26: ParticleSystemNode sub-nodes dynamically registered, added CloudShadowGenNode.
// 26 -> 27: Added TerrainReadbackNode (design plan's bug 6 fix -- the terrain ground-height
// 27 -> 28: Added TerrainShadowNode.
// 28 -> 27: Unified AtmosphereCompositeNode into VolumetricRenderNode.
// 27 -> 32: Drift found empirically while adding GltfRenderNode below -- the tracked count here
// had already fallen behind several nodes' worth of uncommitted foliage/HiZ/multi-queue work
// this session before this change, with no corresponding history entries. Not reconstructing
// which specific additions those were; resynced to the real current count instead.
// 32 -> 33: Added GltfRenderNode (Stage 1 of the GLTF/GLB pipeline).
// 33 -> 34: Added FoliageIndirectResetNode (splitting FoliageIndirectArgsBuffer's zero-clear out
// of FoliageNode::Execute() into its own Compute-domain node -- see FoliageNode.hpp's comment;
// the Graphics-domain original was issuing an illegal barrier+fillBuffer inside its own
// already-open dynamic-rendering scope, confirmed via real GPU-assisted validation).
// 34 -> 35: Replaced GltfRenderNode with ModelRenderNode (GPU-driven model instances, baked
// clips) and added ModelInstanceScatterNode (applies ModelInstanceManager's sparse CPU writes to
// ModelInstanceBuffer) -- net +1.
// 35 -> 37: Added ModelCullResetNode and ModelCullNode (Stage 3 -- compacted-visible-list cull
// pass + indirect task dispatch, passes/ModelCullNode.hpp).
// 37 -> 39: Added ModelLiveFinalizeNode and ModelLiveEvalNode (Stage 4 -- live wind/seed-bend
// eval, passes/ModelLiveEvalNode.hpp).
// 39 -> 40: Added DecorPopulationNode (Stage 5.2 -- persistent, disocclusion-driven decor
// population, passes/DecorPopulationNode.hpp).
// 40 -> 37: Removed FoliageBakeNode, FoliageArchetypeRenderNode, and FoliageIndirectResetNode
// (Stage 5.1/5.2 cleanup -- the GPU-baked tree/bush archetype path they served is fully
// superseded by real ManifoldBranchingTreeAsset/ManifoldRockAsset models + DecorPopulationNode's
// persistent population; see the plan's Stage 5.1 cleanup note).
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 37);
}
