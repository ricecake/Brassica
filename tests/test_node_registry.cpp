#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

// Deliberately requires no Vulkan device at all: registration (render::NodeRegistrar<T>'s static
// initializer, include/render/NodeLifecycle.hpp) is a pure static-init/link-time fact, entirely
// independent of whether a physical device is available -- unlike test_gradient_node.cpp/
// test_rendergraph.cpp, which bootstrap through the real Engine and therefore skip on this Mac
// (MoltenVK lacks the VRS/mesh-shader features Engine::InitVulkan hard-requires), this test can
// give a real answer here.
//
// This is exactly the concrete evidence the CRTP auto-registration plan called for: this
// project builds libbrassica.a as a static library with LTO
// (CMAKE_INTERPROCEDURAL_OPTIMIZATION, see CMakeLists.txt) -- precisely the kind of build where a
// self-registering static with nothing else referencing it can be silently stripped. Including
// Engine.hpp here (which includes passes/AllNodes.hpp) is what forces every node's registrar to
// actually be instantiated and linked into this test's executable; if any were being stripped,
// RegisteredTypeCount() would read back less than 7.
//
// 8 is every top-level node include/passes/AllNodes.hpp lists: GradientNode, TerrainGenNode, TerrainNode,
// DeferredNode, WaterNode, TransmittanceLUTNode, MultiScatteringLUTNode, ParticleSystemNode.
// Particle sub-nodes (ParticleResetNode/LivenessNode/BehaviorNode/RenderNode) are not counted --
// they're constructed and Init'd by ParticleSystemNode itself, not registered with this registry.
// This count should move by exactly one whenever a node is added to or removed from AllNodes.hpp.
TEST_CASE("Every AllNodes.hpp node's CRTP registrar survives static-library linking under this build's LTO") {
	CHECK(brassica::render::EngineNodeRegistry::Instance().RegisteredTypeCount() == 8);
}
