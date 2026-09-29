#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "animation/SkinnedCylinder.hpp"

TEST_CASE("SkinnedCylinder correctly constructs mesh via Geometry Central and skins via Ozz Animation") {
	brassica::SkinnedCylinder::Parameters params;
	params.radius = 1.0f;
	params.height = 4.0f;
	params.radialSegments = 12;
	params.heightSegments = 8;

	brassica::SkinnedCylinder cylinder(params);

	// 1. Verify Geometry Central mesh generation
	CHECK(cylinder.GetSurfaceMesh() != nullptr);
	CHECK(cylinder.GetGeometry() != nullptr);
	CHECK(cylinder.GetVertexCount() > 0);
	CHECK(cylinder.GetTriangleCount() > 0);
	CHECK(cylinder.GetIndices().size() == cylinder.GetTriangleCount() * 3);

	// 2. Verify Ozz Skeleton creation
	CHECK(cylinder.GetSkeleton() != nullptr);
	CHECK(cylinder.GetSkeleton()->num_joints() == 2);

	// Record initial rest/initial skinned positions
	std::vector<glm::vec3> initialPositions = cylinder.GetPositions();
	CHECK(initialPositions.size() == cylinder.GetVertexCount());

	// 3. Update animation pose (time = 1.5s generates rotation on Joint 1)
	cylinder.Update(1.5f);

	std::vector<glm::vec3> animatedPositions = cylinder.GetPositions();
	CHECK(animatedPositions.size() == initialPositions.size());

	// Base vertices (y ~ -2.0) should remain unchanged (joint 0 weight = 1.0, joint 0 rest pose)
	// Top vertices (y ~ +2.0) should shift positions due to joint 1 rotation
	bool foundDifferenceAtTop = false;
	float maxDeltaBase = 0.0f;

	for (std::size_t i = 0; i < initialPositions.size(); ++i) {
		float delta = glm::distance(initialPositions[i], animatedPositions[i]);
		if (initialPositions[i].y > 1.0f) {
			if (delta > 0.01f) {
				foundDifferenceAtTop = true;
			}
		} else if (initialPositions[i].y < -1.9f) {
			maxDeltaBase = std::max(maxDeltaBase, delta);
		}
	}

	CHECK(foundDifferenceAtTop);
	CHECK(maxDeltaBase < 0.001f);
}
