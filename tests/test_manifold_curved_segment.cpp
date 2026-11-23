#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"
#include "passes/model/ModelTypes.hpp"

using namespace brassica;

namespace {
	float MaxAbsX(const ModelAsset& asset) {
		float maxAbsX = 0.0f;
		for (const GltfVertex& v : asset.vertices) {
			maxAbsX = std::max(maxAbsX, std::abs(v.position.x));
		}
		return maxAbsX;
	}

	bool AllFinite(const ModelAsset& asset) {
		for (const GltfVertex& v : asset.vertices) {
			if (!std::isfinite(v.position.x) || !std::isfinite(v.position.y) || !std::isfinite(v.position.z)) {
				return false;
			}
		}
		return true;
	}
} // namespace

TEST_CASE("A joint with no curveWaypoints produces a straight segment (no X-bulge beyond its own radius)") {
	ManifoldJointSpec joint{};
	joint.start = glm::vec3(0.0f, 0.0f, 0.0f);
	joint.end = glm::vec3(0.0f, 4.0f, 0.0f);
	joint.radiusLow = 0.2f;
	joint.radiusHigh = 0.15f;
	joint.parentIndex = kModelNoSkeletonSentinel;
	joint.rigidity = 1.0f;

	ModelAsset asset = BuildSkinnedManifoldAsset(std::span(&joint, 1), 10);
	REQUIRE(!asset.vertices.empty());
	// A straight segment along +Y never strays past its own max radius in X.
	CHECK(MaxAbsX(asset) <= 0.21f);
}

TEST_CASE("A joint with curveWaypoints bends -- real X displacement well beyond the tube's own radius") {
	ManifoldJointSpec joint{};
	joint.start = glm::vec3(0.0f, 0.0f, 0.0f);
	joint.end = glm::vec3(0.0f, 4.0f, 0.0f);
	joint.radiusLow = 0.2f;
	joint.radiusHigh = 0.15f;
	joint.parentIndex = kModelNoSkeletonSentinel;
	joint.rigidity = 0.5f;
	// A gentle bow out to X=1.5 at the midpoint and back -- a real, safe bend (bow depth is large
	// relative to the tube's own 0.2 radius, but spread over the full 4-unit length, so the
	// curvature-vs-radius safety check should accept it).
	joint.curveWaypoints = {glm::vec3(1.5f, 2.0f, 0.0f)};

	ModelAsset asset = BuildSkinnedManifoldAsset(std::span(&joint, 1), 10);
	REQUIRE(!asset.vertices.empty());
	REQUIRE(AllFinite(asset));
	// The bend itself should be visible well past the tube's own ~0.2 radius.
	CHECK(MaxAbsX(asset) > 0.5f);
}

TEST_CASE("A joint with an absurdly tight curve falls back safely -- no crash, no NaN, still a real mesh") {
	ManifoldJointSpec joint{};
	joint.start = glm::vec3(0.0f, 0.0f, 0.0f);
	joint.end = glm::vec3(0.0f, 1.0f, 0.0f);
	joint.radiusLow = 1.0f; // a fat tube
	joint.radiusHigh = 1.0f;
	joint.parentIndex = kModelNoSkeletonSentinel;
	joint.rigidity = 0.5f;
	// A near-180-degree doubling-back within a 1-unit span on a radius-1 tube -- geometrically
	// guaranteed to fold over itself if actually warped; CurveIsSafeToWarp should reject this and
	// fall back to a plain straight segment instead.
	joint.curveWaypoints = {glm::vec3(5.0f, 0.5f, 0.0f)};

	ModelAsset asset = BuildSkinnedManifoldAsset(std::span(&joint, 1), 10);
	REQUIRE(!asset.vertices.empty());
	CHECK(AllFinite(asset));
}

TEST_CASE("Joint attribution survives a curved segment -- trunk and curved branch stay separately tagged") {
	ManifoldJointSpec trunk{};
	trunk.start = glm::vec3(0.0f, 0.0f, 0.0f);
	trunk.end = glm::vec3(0.0f, 2.0f, 0.0f);
	trunk.radiusLow = 0.3f;
	trunk.radiusHigh = 0.25f;
	trunk.parentIndex = kModelNoSkeletonSentinel;
	trunk.rigidity = 1.0f;

	ManifoldJointSpec branch{};
	branch.start = glm::vec3(0.0f, 2.0f, 0.0f);
	branch.end = glm::vec3(1.0f, 4.0f, 0.0f);
	branch.radiusLow = 0.2f;
	branch.radiusHigh = 0.1f;
	branch.parentIndex = 0;
	branch.rigidity = 0.4f;
	branch.curveWaypoints = {glm::vec3(0.8f, 3.0f, 0.0f)};

	std::array<ManifoldJointSpec, 2> joints{trunk, branch};
	ModelAsset                       asset = BuildSkinnedManifoldAsset(joints, 10);

	REQUIRE(asset.jointCount == 2);
	REQUIRE(asset.skeleton.size() == 2);
	REQUIRE(!asset.vertices.empty());

	bool sawJoint0 = false, sawJoint1 = false;
	for (const GltfVertex& v : asset.vertices) {
		if (v.joints.x == 0u) sawJoint0 = true;
		if (v.joints.x == 1u) sawJoint1 = true;
	}
	CHECK(sawJoint0);
	CHECK(sawJoint1);
}
