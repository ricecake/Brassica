#pragma once

#include <cstdint>

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// Stage 5.1 of the GPU-driven model-instance plan: a real recursive branching tree,
	// generalizing Stage 5.0's hand-written 2-joint trunk+branch prototype
	// (BuildManifoldTestTree) via the shared ManifoldSkeletalMeshBuilder pipeline, with real
	// per-branch rigging -- the fidelity confirmed with sgreenhusted over a single rigid sway
	// joint.
	struct TreeGenParams {
		std::uint32_t seed{0};
		int           maxDepth{3};
		int           minBranchesPerNode{2};
		int           maxBranchesPerNode{3};
		float         trunkLength{2.5f};
		float         trunkRadiusLow{0.22f};
		float         trunkRadiusHigh{0.16f};
		float         lengthFalloff{0.65f};  // child length = parent length * this, +/- lengthVariance
		float         lengthVariance{0.15f}; // fraction
		float         radiusFalloff{0.65f};  // child's own taper, continuing from the parent's own end radius
		float         pitchDegreesMin{20.0f};
		float         pitchDegreesMax{55.0f};
		float         baseRigidity{0.95f};           // trunk
		float         rigidityFalloffPerLevel{0.3f}; // more flexible with each level of depth
		int           circularSegments{10};
		// Hard cap, comfortably under kModelLiveMaxJointCount (72) -- ModelCullNode's live-mode
		// eligibility check silently falls back to baked/rest for any model whose jointCount
		// exceeds that budget, so this generator enforces the limit itself rather than relying
		// on maxDepth/branch-count alone (worst case 3 branches/level * depth 3 would be 40).
		// ProceduralArchetypeCatalog's tree/bush presets override this default higher (up to 64)
		// to avoid truncating real trees -- keep that override + 1 (ProceduralPlantGenerator's
		// canopy joint, added on top of this cap, not counted against it) <= kModelLiveMaxJointCount.
		std::uint32_t maxJoints{24};
	};

	ModelAsset BuildProceduralBranchingTree(const TreeGenParams& params);

} // namespace brassica
