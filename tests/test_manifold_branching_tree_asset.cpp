#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "passes/model/ManifoldBranchingTreeAsset.hpp"
#include "passes/model/ModelRegistry.hpp"

using namespace brassica;

TEST_CASE("BuildProceduralBranchingTree produces a real, rigged, bounded-size skinned mesh") {
	TreeGenParams params;
	params.seed = 7;
	ModelAsset tree = BuildProceduralBranchingTree(params);

	REQUIRE(!tree.vertices.empty());
	REQUIRE(!tree.meshlets.empty());
	REQUIRE(tree.jointCount > 1); // a real branching structure, not just the trunk alone

	// Hard joint cap -- ModelCullNode's live-mode eligibility silently falls back to baked/rest
	// for any model exceeding kModelLiveMaxJointCount (32); this generator must never produce
	// more joints than that, with real margin.
	CHECK(tree.jointCount <= 24);
	REQUIRE(tree.skeleton.size() == tree.jointCount);

	// Topological order (parentIndex < this joint's own index, or the root sentinel), the
	// contract ModelLiveEvalNode's serial joint walk relies on.
	CHECK(tree.skeleton[0].parentIndex == kModelNoSkeletonSentinel);
	for (std::uint32_t j = 1; j < tree.jointCount; ++j) {
		CHECK(tree.skeleton[j].parentIndex < j);
	}

	// Rigidity tapers from a rigid trunk toward flexible branch tips -- a leaf joint (one with
	// no children) should never be stiffer than its own parent.
	std::vector<bool> hasChild(tree.jointCount, false);
	for (std::uint32_t j = 1; j < tree.jointCount; ++j) {
		hasChild[tree.skeleton[j].parentIndex] = true;
	}
	for (std::uint32_t j = 1; j < tree.jointCount; ++j) {
		CHECK(tree.skeleton[j].rigidity <= tree.skeleton[tree.skeleton[j].parentIndex].rigidity);
	}
	CHECK(tree.skeleton[0].rigidity > 0.9f); // trunk stays close to rigid

	for (const GltfVertex& v : tree.vertices) {
		float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
		CHECK(weightSum == doctest::Approx(1.0f));
		CHECK(v.joints.x < tree.jointCount);
		CHECK(std::isfinite(v.position.x));
		float normalLen = glm::length(v.normal);
		CHECK(normalLen > 0.9f);
		CHECK(normalLen < 1.1f);
	}

	// Every joint actually shows up in the final mesh -- confirms OriginalID attribution
	// worked across the whole branching structure, not just the first couple of joints.
	std::vector<bool> jointSeen(tree.jointCount, false);
	for (const GltfVertex& v : tree.vertices) {
		jointSeen[v.joints.x] = true;
	}
	for (std::uint32_t j = 0; j < tree.jointCount; ++j) {
		CHECK(jointSeen[j]);
	}

	CHECK(tree.restBoundsRadius > 0.0f);
	CHECK(tree.palette.size() == tree.jointCount);

	ModelRegistry registry;
	std::uint32_t modelIndex = registry.RegisterAsset(tree);
	CHECK(registry.GetJointCount(modelIndex) == tree.jointCount);
}

TEST_CASE("BuildProceduralBranchingTree stays within the joint cap across several seeds") {
	for (std::uint32_t seed = 0; seed < 20; ++seed) {
		TreeGenParams params;
		params.seed = seed;
		ModelAsset tree = BuildProceduralBranchingTree(params);
		CHECK(tree.jointCount <= 24);
		CHECK(tree.jointCount >= 1);
	}
}
