#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <manifold/manifold.h>

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

		// Procedural plant generation v2: intermediate waypoints (absolute rest-space, same space
		// as start/end) a curved segment should pass through, in order from start to end. Empty
		// (the default) means a straight segment -- byte-identical to every joint built before
		// this field existed, since the construction code below only takes the curved path when
		// this is non-empty. See ManifoldSkeletalMeshBuilder.cpp's own comment on how a straight
		// tapered Manifold::Extrude gets bent along this path via Warp + a parallel-transport
		// frame, and what happens if the bend is tight enough to self-intersect.
		std::vector<glm::vec3> curveWaypoints;

		// Procedural plant generation v2: if set, replaces the default straight/curved-cylinder
		// construction for this joint entirely -- the callback must return a Manifold in the same
		// canonical local frame a straight segment would occupy (base at local z=0, growing to
		// +Z, same convention Manifold::Cylinder itself uses), which still gets positioned via the
		// normal start/end transform below it. start/end/parentIndex/rigidity still govern the
		// skeleton bone's own transform regardless of what geometry this produces -- e.g. a flower
		// head (ManifoldFlowerAsset.hpp) is a real joint via this hook, not a separate attachment
		// concept, so it inherits rigidity/wind-bend for free. Ignored if curveWaypoints is also
		// set -- a joint is either a custom shape or a (possibly curved) tube, not both.
		std::function<manifold::Manifold()> customShape;
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
