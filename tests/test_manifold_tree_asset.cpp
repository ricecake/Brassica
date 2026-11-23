#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "passes/model/ManifoldTreeAsset.hpp"
#include "passes/model/ModelRegistry.hpp"

using namespace brassica;

TEST_CASE("BuildManifoldTestTree produces a real, internally consistent skinned mesh") {
	ModelAsset tree = BuildManifoldTestTree();

	// A real mesh came out of the Manifold union -- not the empty fallback the
	// generator returns on a boolean-union failure.
	REQUIRE(!tree.vertices.empty());
	REQUIRE(!tree.meshlets.empty());
	REQUIRE(!tree.meshletVertices.empty());
	REQUIRE(!tree.meshletTrianglesPacked.empty());

	CHECK(tree.jointCount == 2);
	REQUIRE(tree.skeleton.size() == tree.jointCount);

	// Topological order (parentIndex < this joint's own index), same contract
	// ModelLiveEvalNode's serial joint walk relies on.
	CHECK(tree.skeleton[0].parentIndex == kModelNoSkeletonSentinel);
	CHECK(tree.skeleton[1].parentIndex == 0);
	CHECK(tree.skeleton[0].rigidity > tree.skeleton[1].rigidity); // trunk rigid, branch flexible

	for (const GltfVertex& v : tree.vertices) {
		// Rigid single-joint skinning, same convention ProceduralTreeAsset uses.
		float weightSum = v.weights.x + v.weights.y + v.weights.z + v.weights.w;
		CHECK(weightSum == doctest::Approx(1.0f));
		CHECK(v.joints.x < tree.jointCount);

		// No NaN/garbage from the Manifold union or normal calculation.
		CHECK(std::isfinite(v.position.x));
		CHECK(std::isfinite(v.position.y));
		CHECK(std::isfinite(v.position.z));
		CHECK(std::isfinite(v.normal.x));
		CHECK(std::isfinite(v.normal.y));
		CHECK(std::isfinite(v.normal.z));
		// Real, roughly-unit-length normal, not the all-zero degenerate vector a wrong
		// CalculateNormals()/GetMeshGL() property-channel index silently produces (a real,
		// hardware-tested bug this guards against: CalculateNormals(3) + GetMeshGL(3) reads
		// unused zero padding instead of the real normal Manifold wrote at a different offset
		// -- see BuildManifoldTestTree's own comment).
		float normalLen = glm::length(v.normal);
		CHECK(normalLen > 0.9f);
		CHECK(normalLen < 1.1f);
	}

	// Both joints are actually represented in the final mesh -- a real check that
	// OriginalID/runOriginalID attribution worked, not just that joint 0 claimed
	// everything by accident (e.g. an empty idToJoint lookup silently defaulting
	// every vertex to 0).
	bool sawJoint0 = false, sawJoint1 = false;
	for (const GltfVertex& v : tree.vertices) {
		if (v.joints.x == 0u) sawJoint0 = true;
		if (v.joints.x == 1u) sawJoint1 = true;
	}
	CHECK(sawJoint0);
	CHECK(sawJoint1);

	CHECK(tree.restBoundsRadius > 0.0f);
	CHECK(tree.palette.size() == tree.jointCount);

	// Registers cleanly through the unchanged ModelRegistry path -- no new mechanism
	// needed, per the plan's own redirect.
	ModelRegistry registry;
	std::uint32_t modelIndex = registry.RegisterAsset(tree);
	CHECK(modelIndex == 0);
	CHECK(registry.GetJointCount(modelIndex) == tree.jointCount);
	CHECK(registry.GetMeshletCount(modelIndex) == tree.meshlets.size());
}
