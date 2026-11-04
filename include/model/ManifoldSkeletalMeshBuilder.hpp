#pragma once

#include <cstdint>
#include <span>

#include <glm/glm.hpp>

#include "model/ModelAsset.hpp"

namespace brassica {

	// One joint's rigid tapered-cylinder segment, in absolute rest-space (not parent-relative --
	// same "inverse-bind skinning needs vertices positioned where they already look right at
	// rest" convention Stage 2's ProceduralTreeAsset and Stage 5.0's BuildManifoldTestTree
	// already use). joints must be in topological order (parentIndex < this joint's own index
	// within the span).
	struct ManifoldJointSpec {
		glm::vec3     start;
		glm::vec3     end;
		float         radiusLow;
		float         radiusHigh;
		std::uint32_t parentIndex; // kModelNoSkeletonSentinel for the root
		float         rigidity;   // 0 = fully flexible, 1 = rigid (no wind/seed bend)
	};

	// Shared Manifold-CSG-to-skinned-ModelAsset pipeline, extracted from Stage 5.0's
	// BuildManifoldTestTree (the GPU-driven model-instance plan's proof of concept) so every
	// skeletal archetype generator (trees, bushes) uses this exactly once instead of
	// reimplementing it: one tapered Manifold::Cylinder per joint, unioned via CSG, with exact
	// per-vertex joint attribution recovered from Manifold's own mesh-relation tracking
	// (OriginalID/AsOriginal/runOriginalID -- not interpolated properties or a geometric
	// nearest-point heuristic, see the plan's own Stage 5.0 section for why), real meshlets via
	// the shared MeshletBuilder, and a skeleton built the same TRS-composition way
	// ProceduralTreeAsset/BuildManifoldTestTree already do. Returns an empty-geometry ModelAsset
	// (jointCount still set) if the CSG union fails -- callers get a harmless no-op model rather
	// than a crash, same "failed load leaves an empty asset" convention GltfModelManager uses.
	ModelAsset BuildSkinnedManifoldAsset(
		std::span<const ManifoldJointSpec> joints, int circularSegments = 12, std::uint32_t materialIndex = 0u
	);

} // namespace brassica
