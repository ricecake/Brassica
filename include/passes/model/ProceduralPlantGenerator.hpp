#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"
#include "passes/model/ModelAsset.hpp"

namespace brassica {

	// Procedural plant generation v2: grows a PlantIR (short, slightly-jittered chained
	// segments, same recursive-branching spirit as TreeGenParams/BuildProceduralBranchingTree),
	// optimizes it (ProceduralPlantOptimizer -- consolidates each chain into one curved joint,
	// flares real branch points into Hubs), then converts the result into ManifoldJointSpecs
	// (curved per consolidated chain, flared at hubs, a real flower head at flagged terminal
	// tips when enabled) and calls BuildSkinnedManifoldAsset. Replaces
	// BuildProceduralBranchingTree's role for Tree/Bush archetypes -- strictly more capable
	// (curved branches, real junction flares, flower support); ManifoldRockAsset is untouched,
	// rocks aren't articulated.
	struct PlantGenParams {
		std::uint32_t seed{0};

		// Growth: each step is one short segment, not one branch level -- a run of
		// single-child, slightly-jittered steps is what the optimizer later consolidates into one
		// smoothly curved joint, so a visually "simple" branch is actually several of these.
		int   maxSegments{40}; // raw segment budget before optimization/consolidation
		float segmentLength{0.4f};
		float segmentLengthVariance{0.2f};  // fraction
		float directionJitterDegrees{12.0f}; // per-segment continuation wander -- what makes a
											  // consolidated chain curve instead of staying straight
		float branchChance{0.35f}; // probability a given step branches instead of continuing
		int   minBranchesPerNode{2};
		int   maxBranchesPerNode{3};
		float branchPitchMinDegrees{20.0f};
		float branchPitchMaxDegrees{55.0f};
		// A freshly-grown branch child must grow this many plain continuation segments before
		// it's itself eligible to branch again. Without this, a branch's own children can branch
		// again on the very next step, stacking a second branch point directly against the first
		// one (no straight run of tube between them) -- the mechanical cause of the "crowded"/
		// colliding branch clusters seen at high branchChance (BushVariantA/B's 0.5). 0 restores
		// the old no-cooldown behavior.
		int minSegmentsBetweenBranches{1};

		float trunkRadiusLow{0.22f};
		float trunkRadiusHigh{0.16f};
		float radiusFalloffPerSegment{0.93f}; // gentle -- segments are short, falloff compounds

		float baseRigidity{0.9f};
		float rigidityFalloffPerSegment{0.025f};

		int circularSegments{10};
		// Hard cap on the FINAL (post-consolidation) joint count, enforced while converting the
		// optimized IR into ManifoldJointSpecs -- comfortably under kModelLiveMaxJointCount (32),
		// same reasoning TreeGenParams::maxJoints already documents.
		std::uint32_t maxJoints{24};

		// Flowers: a terminal (leaf) segment has this chance of growing a real flower head
		// (ManifoldFlowerAsset.hpp) instead of just ending bare.
		bool  enableFlowers{false};
		float flowerChance{0.5f};
		float flowerSize{0.25f};

		// Leaves: a terminal tip that didn't already grow a flower has this chance of growing a
		// small tuft of leaf blades (ManifoldLeafAsset.hpp) instead of just ending bare.
		bool          enableLeaves{false};
		float         leafChance{0.7f};
		float         leafSize{0.12f};
		int           leafCount{5};
		float         leafWidthRatio{0.45f}; // see BuildLeafCluster's own comment -- smaller reads as a frond
		// true: BuildLeafBallCluster (overlapping balls + blades stuck to the side, a real
		// canopy clump). false: the plain BuildLeafCluster fan (blades splayed from one shared
		// point) -- Palm sets this false, since a ball cluster reads as foliage mass, not the
		// single wide frond crown Palm's own leafWidthRatio/leafCount/leafSize are tuned for.
		bool leafBallCluster{true};
		// If set, leaf-cluster joints use this material instead of materialIndex below (e.g.
		// green leaves on an otherwise bark-brown tree) -- see ManifoldJointSpec::materialIndex.
		std::optional<std::uint32_t> leafMaterialIndex;

		std::uint32_t materialIndex{0}; // caller's concern, same contract as BuildMeshlets' own
	};

	ModelAsset BuildProceduralPlant(const PlantGenParams& params);

	// The growth+optimize+convert pipeline alone, stopping just before BuildSkinnedManifoldAsset
	// -- a testable seam so curveWaypoints/customShape joints (Hubs, flowers) can be inspected
	// directly instead of only through their indirect effect on a finished ModelAsset's
	// vertices. BuildProceduralPlant is just this followed by BuildSkinnedManifoldAsset.
	std::vector<ManifoldJointSpec> BuildPlantJoints(const PlantGenParams& params);

} // namespace brassica
