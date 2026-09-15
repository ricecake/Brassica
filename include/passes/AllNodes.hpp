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

#include "passes/AtmosphereLUTNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/GradientNode.hpp"
#include "passes/ParticleSystemNode.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"
