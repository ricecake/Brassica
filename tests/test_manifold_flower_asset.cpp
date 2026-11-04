#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "model/ManifoldFlowerAsset.hpp"
#include "model/ManifoldSkeletalMeshBuilder.hpp"
#include "model/ModelTypes.hpp"

using namespace brassica;

TEST_CASE("BuildFlowerHead produces a valid, correctly-wound closed manifold for every profile") {
	for (FlowerProfile profile : {FlowerProfile::Diamond, FlowerProfile::Bell, FlowerProfile::Star, FlowerProfile::Heart}) {
		manifold::Manifold head = BuildFlowerHead(0.3f, profile);
		REQUIRE(head.Status() == manifold::Manifold::Error::NoError);
		CHECK(head.NumVert() == 10);
		CHECK(head.NumTri() == 16);
		// A positive volume confirms outward-facing (correct CCW) winding on both the top and
		// bottom fans -- inverted winding on either would make this zero or negative.
		CHECK(head.Volume() > 0.0);
	}
}

TEST_CASE("A flower head plugged in as a joint's customShape produces a real, separately-attributed terminal joint") {
	ManifoldJointSpec stem{};
	stem.start = glm::vec3(0.0f, 0.0f, 0.0f);
	stem.end = glm::vec3(0.0f, 2.0f, 0.0f);
	stem.radiusLow = 0.08f;
	stem.radiusHigh = 0.05f;
	stem.parentIndex = kModelNoSkeletonSentinel;
	stem.rigidity = 0.3f;

	ManifoldJointSpec flowerJoint{};
	flowerJoint.start = glm::vec3(0.0f, 2.0f, 0.0f);
	flowerJoint.end = glm::vec3(0.0f, 2.05f, 0.0f); // tiny non-zero length, just to define an orientation
	flowerJoint.parentIndex = 0;
	flowerJoint.rigidity = 1.0f; // rigid relative to the stem tip, same convention as a rock
	flowerJoint.customShape = [] { return BuildFlowerHead(0.25f, FlowerProfile::Star); };

	std::array<ManifoldJointSpec, 2> joints{stem, flowerJoint};
	ModelAsset                       asset = BuildSkinnedManifoldAsset(joints, 10);

	REQUIRE(asset.jointCount == 2);
	REQUIRE(!asset.vertices.empty());

	bool sawStem = false, sawFlower = false;
	for (const GltfVertex& v : asset.vertices) {
		if (v.joints.x == 0u) sawStem = true;
		if (v.joints.x == 1u) sawFlower = true;
	}
	CHECK(sawStem);
	CHECK(sawFlower);
}
