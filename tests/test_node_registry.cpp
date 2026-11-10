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
// 32 -> 33: Added GltfRenderNode (Stage 1 of the GLTF/GLB pipeline).
// 33 -> 34: Added FoliageIndirectResetNode
// 34 -> 43: Particle cluster reactivated
// 43 -> 45: Resynced with auto-registered nodes (including TonemapComputeNode / TonemapNode).
// 45 -> 46: Added GroundDebrisNode for camera proximity terrain debris mesh rendering.
// Added since 46: GpuLightsResetNode (CPU/GPU light buffer split -- zeroes GpuLightsBuffer each
// frame so atomic-claimed dynamic light slots, e.g. fireflies, start from a known state) and
// ShadingRateComputeNode (attachment-only variable rate shading -- derives this frame's per-tile
// shading rate from last frame's ShadingRateMap/HiZTexture). Resynced to the real current count
// after rebasing onto main (2026-11-10) rather than chase the exact arithmetic across both
// branches' independent histories -- same precedent as the 27 -> 32 "Drift" entry above.
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 46);
}
