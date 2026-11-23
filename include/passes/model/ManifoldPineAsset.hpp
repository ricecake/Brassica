#pragma once

#include <cstdint>
#include <optional>

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// A dedicated, bespoke pine generator -- NOT a parametrization of
	// ProceduralPlantGenerator's general branching-chain pipeline. sgreenhusted confirmed the
	// branching-chain approach (narrow branchPitch window trying to fake a conifer silhouette
	// out of the same mechanism trees/bushes use) was fundamentally the wrong shape and asked
	// for a real port of the pre-conversion look instead: the old GPU-emitted
	// `emitPineTreeItem` (shaders/foliage.mesh, deleted when the GPU-driven-model/Manifold
	// conversion landed, see git history around commit 676d60d) built a short, simple,
	// mostly-straight trunk topped with 3 stacked cone-shaped foliage tiers (decreasing radius
	// going up) -- not branches at all. This ports that same silhouette: a plain tapered trunk
	// (no branching, no curve-safety exposure -- same reasoning Palm's own branchChance=0 trunk
	// already uses) plus `tierCount` cone tiers (`manifold::Manifold::Cylinder` with
	// radiusHigh=0 is exactly a cone -- no custom mesh-building needed), each tier a child of
	// the one below it for cumulative wind sway, going needle-green via materialIndex.
	struct PineGenParams {
		std::uint32_t seed{0};

		int   trunkSegments{4};
		float trunkHeight{2.6f};
		float trunkRadiusLow{0.16f};
		float trunkRadiusHigh{0.09f};
		float trunkBaseRigidity{0.95f};

		int   tierCount{3};
		float tierBaseRadius{0.85f};     // the lowest (widest) tier's own base radius
		float tierRadiusFalloff{0.70f}; // each higher tier's radius relative to the one below
		float tierHeight{1.15f};         // each cone's own height
		float tierOverlap{0.35f};        // fraction of tierHeight consecutive tiers overlap by
		float tierBaseRigidity{0.55f};   // lower than the trunk -- foliage sways more

		int circularSegments{10};

		std::uint32_t materialIndex{0}; // trunk/bark
		// If set, tier (needle) joints use this material instead of materialIndex above --
		// same per-joint override ManifoldJointSpec::materialIndex already provides for Tree/
		// Bush leaves.
		std::optional<std::uint32_t> needleMaterialIndex;
	};

	ModelAsset BuildPineTree(const PineGenParams& params);

} // namespace brassica
