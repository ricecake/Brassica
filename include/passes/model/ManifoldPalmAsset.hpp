#pragma once

#include <cstdint>
#include <optional>

#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// A dedicated, bespoke palm generator -- NOT a parametrization of
	// ProceduralPlantGenerator's general branching-chain pipeline, same reasoning
	// ManifoldPineAsset.hpp's own comment gives for Pine. sgreenhusted clarified (after the
	// branching-chain Palm preset shipped) that the real ask was an *upgraded* port of the
	// pre-conversion look -- the old GPU-emitted `emitPalmTreeItem` (shaders/foliage_emit.glsl,
	// removed by commit dd4478871274e88ad87b18986978186fd25ed0f2) -- not a replacement shape:
	// that old palm already had a recognizable gently-curving trunk and
	// long drooping fronds, but the trunk was a flat 2-vertex-wide ribbon (2D), and the
	// branching-chain preset's own leaf-tuft read as a diminutive clump next to real fronds.
	// This ports the same silhouette -- one quadratically-bending trunk (matching
	// emitPalmTreeItem's own `bend = t*t*0.8` lean, now a real round tube via curveWaypoints
	// instead of a flat ribbon) topped with frondCount long, tapering, drooping fronds (matching
	// its own `dropY = -ft*ft*0.9*frondLen` parabolic droop) -- rather than trying to fake a palm
	// out of the same branch-pitch mechanism trees/bushes use.
	struct PalmGenParams {
		std::uint32_t seed{0};

		int   trunkSegments{5}; // curveWaypoints resolution, not separate joints -- see BuildPalmTree
		float trunkHeight{7.0f};
		float trunkRadiusLow{0.26f};
		float trunkRadiusHigh{0.17f};
		float trunkBendAmount{0.8f}; // matches emitPalmTreeItem's own literal bend coefficient
		float trunkBaseRigidity{0.9f};

		int   frondCount{5};
		float frondLength{2.7f};
		float frondWidth{0.5f};
		float frondDroop{0.9f}; // matches emitPalmTreeItem's own literal dropY coefficient
		int   frondSegments{4}; // chained-blade tessellation of each frond's curve -- see BuildPalmFrond
		float frondRigidity{0.5f};

		int circularSegments{10};

		std::uint32_t materialIndex{0}; // trunk/bark
		// If set, frond joints use this material instead of materialIndex above -- same
		// per-joint override ManifoldJointSpec::materialIndex already provides for Tree/Bush
		// leaves and Pine's own needle tiers.
		std::optional<std::uint32_t> frondMaterialIndex;
	};

	ModelAsset BuildPalmTree(const PalmGenParams& params);

} // namespace brassica
