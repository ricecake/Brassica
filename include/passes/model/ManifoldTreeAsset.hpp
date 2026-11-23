#pragma once

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// Stage 5.0 of the GPU-driven model-instance plan (/Users/sgreenhusted/.claude/plans/
	// fluttering-rolling-puffin.md): proves the Manifold (CSG) -> meshoptimizer ->
	// ModelAsset -> ModelRegistry pipeline end to end on one simple trunk+branch shape,
	// before authoring the real archetype set (Stage 5.1). Unlike ProceduralTreeAsset's
	// hand-rolled cylinder-ring geometry, this builds each joint's segment as a real
	// manifold (Manifold::Cylinder), unions them via CSG, and recovers per-vertex joint
	// attribution from Manifold's own mesh-relation tracking (OriginalID/AsOriginal/
	// runOriginalID) rather than interpolated properties or a geometric nearest-point
	// heuristic -- see the plan's own Stage 5.0 section for why.
	ModelAsset BuildManifoldTestTree();

} // namespace brassica
