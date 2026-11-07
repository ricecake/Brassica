#pragma once

// The one manual touch point auto-registration doesn't eliminate: every engine-level node's
// header must be #included from *some* translation unit that's unconditionally linked, or its
// render::NodeRegistrar<T> static initializer can be stripped by the linker before it ever runs.
// This project builds libbrassica.a as a static library with LTO
// (CMAKE_INTERPROCEDURAL_OPTIMIZATION, see CMakeLists.txt) -- exactly the kind of build where an
// unreferenced translation unit's static side effects are fair game to eliminate. Engine.cpp
// includes this header once (Engine.cpp is always linked, since Engine itself is used
// everywhere), which is what keeps every node's registration reachable. Adding a new node means
// adding one #include line here -- nothing else in Engine.hpp/.cpp needs to change.

#include "passes/atmosphere/nodes/AtmosphereLUTNode.hpp"
#include "passes/clouds/nodes/CirrusNode.hpp"
#include "passes/clouds/nodes/CloudCompositeNode.hpp"
#include "passes/clouds/nodes/CloudRenderNode.hpp"
#include "passes/clouds/nodes/CloudShadowGenNode.hpp"
#include "passes/clouds/nodes/CloudTemporalNode.hpp"
#include "passes/clouds/nodes/CloudVolumeDownsampleNode.hpp"
#include "passes/clouds/nodes/CloudVolumeGenNode.hpp"
#include "passes/lighting/nodes/ClusterLightAssignmentNode.hpp"
#include "passes/model/nodes/DecorPopulationNode.hpp"
#include "passes/scene/nodes/DeferredNode.hpp"
#include "passes/entity/nodes/EntityNode.hpp"
#include "passes/entity/nodes/EntityPrepareNode.hpp"
#include "passes/foliage/nodes/FoliageNode.hpp"
#include "passes/lighting/nodes/GTAONode.hpp"
#include "passes/scene/nodes/HiZDownsampleNode.hpp"
#include "passes/ui/nodes/ImGuiNode.hpp"
#include "passes/lighting/nodes/LightningAgeNode.hpp"
#include "passes/lighting/nodes/LightningRenderNode.hpp"
#include "passes/model/nodes/ModelCullNode.hpp"
#include "passes/model/nodes/ModelInstanceScatterNode.hpp"
#include "passes/model/nodes/ModelLiveEvalNode.hpp"
#include "passes/model/nodes/ModelRenderNode.hpp"
// Particle node cluster: reactivated (S2) -- every node below is now BRASSICA_REGISTER_NODE'd,
// gated at runtime by IParticleManager::IsEnabled() (see passes/particle/resources.hpp's header
// comment).
#include "passes/particle/nodes/AboveWaterParticleRenderNode.hpp"
#include "passes/particle/nodes/ParticleBehaviorNode.hpp"
#include "passes/particle/nodes/ParticleGridBuildNode.hpp"
#include "passes/particle/nodes/ParticleLivenessNode.hpp"
#include "passes/particle/nodes/ParticleResetNode.hpp"
#include "passes/particle/nodes/ParticleTypeBufferNode.hpp"
#include "passes/particle/nodes/UnderwaterParticleRenderNode.hpp"
#include "passes/atmosphere/nodes/SkyBackgroundNode.hpp"
#include "passes/biome/nodes/TerrainBiomeNode.hpp"
#include "passes/terrain/nodes/TerrainGenNode.hpp"
#include "passes/terrain/nodes/TerrainMinMaxDownsampleNode.hpp"
#include "passes/terrain/nodes/TerrainNode.hpp"
#include "passes/terrain/nodes/TerrainReadbackNode.hpp"
#include "passes/terrain/nodes/TerrainShadowNode.hpp"
#include "passes/postprocess/nodes/TonemapNode.hpp"
#include "passes/volumetrics/nodes/VolumetricFroxelGenNode.hpp"
#include "passes/volumetrics/nodes/VolumetricRenderNode.hpp"
#include "passes/water/nodes/WaterNode.hpp"
