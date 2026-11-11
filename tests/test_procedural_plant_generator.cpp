#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "passes/model/ProceduralPlantGenerator.hpp"

using namespace brassica;

namespace {
	bool AllFinite(const ModelAsset& asset) {
		for (const GltfVertex& v : asset.vertices) {
			if (!std::isfinite(v.position.x) || !std::isfinite(v.position.y) || !std::isfinite(v.position.z)) {
				return false;
			}
		}
		return true;
	}
} // namespace

TEST_CASE("BuildProceduralPlant produces a real, rigged, bounded-size skinned mesh across many seeds") {
	for (std::uint32_t seed = 0; seed < 20; ++seed) {
		PlantGenParams params;
		params.seed = seed;

		ModelAsset plant = BuildProceduralPlant(params);

		REQUIRE(!plant.vertices.empty());
		REQUIRE(!plant.meshlets.empty());
		REQUIRE(plant.jointCount > 1);
		CHECK(plant.jointCount <= 24);
		REQUIRE(plant.skeleton.size() == plant.jointCount);
		CHECK(AllFinite(plant));

		// Topological order, the contract ModelLiveEvalNode's serial joint walk relies on.
		CHECK(plant.skeleton[0].parentIndex == kModelNoSkeletonSentinel);
		for (std::uint32_t j = 1; j < plant.jointCount; ++j) {
			CHECK(plant.skeleton[j].parentIndex < j);
		}

		// Every vertex's joint index must be real -- but NOT every joint needs a surviving
		// vertex: a Hub's flare sphere sits exactly at a junction and can end up entirely inside
		// the CSG union of its own abutting tubes, contributing zero visible surface (a real,
		// harmless outcome confirmed empirically, not a bug -- see ProceduralPlantOptimizer's own
		// comment on the flare radius). The joint still exists correctly in the skeleton either
		// way; it just has nothing of its own to render that frame.
		for (const GltfVertex& v : plant.vertices) {
			REQUIRE(v.joints.x < plant.jointCount);
		}
	}
}

TEST_CASE("BuildProceduralPlant with leaves enabled (ball-cluster canopy) produces a valid, bounded mesh") {
	for (std::uint32_t seed = 0; seed < 20; ++seed) {
		PlantGenParams params;
		params.seed = seed;
		params.enableLeaves = true;
		params.leafChance = 1.0f; // force every terminal segment to grow a leaf, for a real exercise
		// leafBallCluster defaults true -- this is the Tree/Bush production path. Palm no longer
		// goes through BuildProceduralPlant at all (see ManifoldPalmAsset.hpp), but the
		// leafBallCluster=false override this field exists for is still a real, available
		// capability for any future preset that wants the plain fan instead.

		ModelAsset plant = BuildProceduralPlant(params);

		REQUIRE(!plant.vertices.empty());
		CHECK(plant.jointCount <= 24);
		CHECK(AllFinite(plant));
	}
}

TEST_CASE("BuildProceduralPlant with flowers enabled still produces a valid, bounded mesh") {
	for (std::uint32_t seed = 0; seed < 10; ++seed) {
		PlantGenParams params;
		params.seed = seed;
		params.enableFlowers = true;
		params.flowerChance = 1.0f; // force every terminal segment to flower, for a real exercise

		ModelAsset plant = BuildProceduralPlant(params);

		REQUIRE(!plant.vertices.empty());
		CHECK(plant.jointCount <= 24);
		CHECK(AllFinite(plant));
	}
}

TEST_CASE("BuildPlantJoints actually produces curved joints via consolidation, not just straight segments") {
	bool sawAnyCurve = false;
	bool sawAnyHub = false;
	for (std::uint32_t seed = 0; seed < 20; ++seed) {
		PlantGenParams params;
		params.seed = seed;
		params.directionJitterDegrees = 20.0f; // a generous wander, to make curvature likely

		std::vector<ManifoldJointSpec> joints = BuildPlantJoints(params);
		REQUIRE(!joints.empty());
		for (const ManifoldJointSpec& j : joints) {
			if (!j.curveWaypoints.empty()) {
				sawAnyCurve = true;
			}
			// A Hub-derived joint has no real radiusLow/radiusHigh of its own (it's a sphere via
			// customShape) -- the clearest external signal a branch flare happened.
			if (j.customShape && j.radiusLow == 0.0f && j.radiusHigh == 0.0f) {
				sawAnyHub = true;
			}
		}
	}
	// If literally no seed out of 20 (with a generous jitter) ever produces a joint with real
	// curve waypoints or a flared branch junction, the optimizer/consolidation step isn't
	// actually doing anything -- this is the direct check the earlier bounding-box-extent signal
	// couldn't actually isolate (branching alone spreads the X/Z extent regardless of whether any
	// individual joint curves).
	CHECK(sawAnyCurve);
	CHECK(sawAnyHub);
}

// Regression test for a real crowding bug: without a cooldown, a branch's own children were
// eligible to branch again on the very next step, stacking a second branch point directly
// against the first with no straight tube run between them -- sgreenhusted reported this as a
// visually crowded tree/bush, and a direct 100-seed sweep across every Tree/Bush preset showed
// it compounding into real self-intersection (BuildCurvedSegment straight-fallback warnings).
// branchChance=1.0 makes the branch/no-branch decision itself deterministic (every eligible step
// branches), isolating minSegmentsBetweenBranches's own effect on hub count from the RNG that
// normally drives that decision.
TEST_CASE("minSegmentsBetweenBranches cooldown strictly reduces how many branch points a fixed segment budget produces") {
	auto countHubs = [](int cooldown) {
		PlantGenParams p;
		p.seed = 42;
		p.maxSegments = 40;
		p.branchChance = 1.0f;
		p.minBranchesPerNode = 2;
		p.maxBranchesPerNode = 2;
		p.minSegmentsBetweenBranches = cooldown;

		std::vector<ManifoldJointSpec> joints = BuildPlantJoints(p);
		int                            hubCount = 0;
		for (const ManifoldJointSpec& j : joints) {
			if (j.customShape && j.radiusLow == 0.0f && j.radiusHigh == 0.0f) {
				++hubCount;
			}
		}
		return hubCount;
	};

	CHECK(countHubs(3) < countHubs(0));
}
