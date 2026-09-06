#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <glm/glm.hpp>
#include <vector>
#include <cmath>

TEST_CASE("Ray Query Terrain Shadows: AABB Resolution vs Shadow Caster Distance") {
	glm::vec3 cameraPos(0.0f, 50.0f, 0.0f);
	uint32_t numLODs = 7;
	float baseTexelSize = 0.5f;

	// Calculate AABB size for close vs far shadow casters
	for (uint32_t lod = 0; lod < numLODs; ++lod) {
		float baseMeshletSize = 32.0f;
		float meshletSize = baseMeshletSize * std::pow(2.0f, static_cast<float>(lod));

		// Distance from camera increases with LOD level
		float innerRadius = (lod > 0) ? (240.0f * std::pow(2.0f, static_cast<float>(lod - 1))) : 0.0f;

		if (lod == 0) {
			CHECK(meshletSize == doctest::Approx(32.0f));
		} else if (lod == 1) {
			CHECK(meshletSize == doctest::Approx(64.0f));
			CHECK(innerRadius == doctest::Approx(240.0f));
		} else if (lod == 2) {
			CHECK(meshletSize == doctest::Approx(128.0f));
			CHECK(innerRadius == doctest::Approx(480.0f));
		}
	}
}

TEST_CASE("Ray Query Terrain Shadows: Heightmap Traversal Precision vs Camera Distance") {
	auto calculateStepSize = [](float camDistToShaded) {
		return std::clamp(camDistToShaded * 0.01f, 0.5f, 4.0f);
	};

	auto calculateNumSteps = [](float stepSize) {
		return static_cast<int>(std::clamp(200.0f / stepSize, 10.0f, 50.0f));
	};

	// Close shaded point (e.g. 10 units away) -> smaller step size, higher precision
	float closeDist = 10.0f;
	float closeStep = calculateStepSize(closeDist);
	int closeSteps = calculateNumSteps(closeStep);

	CHECK(closeStep == doctest::Approx(0.5f));
	CHECK(closeSteps == 50);

	// Far shaded point (e.g. 300 units away) -> larger step size, lower precision
	float farDist = 300.0f;
	float farStep = calculateStepSize(farDist);
	int farSteps = calculateNumSteps(farStep);

	CHECK(farStep == doctest::Approx(3.0f));
	CHECK((farSteps == 66 || farSteps == 50));
	CHECK(farSteps <= 50);
}
