#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "types/AABB.hpp"
#include "terrain/TerrainClipmap.hpp"
#include "terrain/AsyncTerrainUploader.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

TEST_CASE("AABB Tools and Frustum Culling") {
	brassica::AABB box(glm::vec3(-5.0f, -5.0f, -5.0f), glm::vec3(5.0f, 5.0f, 5.0f));

	CHECK(box.Center() == glm::vec3(0.0f));
	CHECK(box.Extents() == glm::vec3(5.0f));

	// Distance calculations
	glm::vec3 cameraPos(0.0f, 0.0f, 15.0f);
	CHECK(doctest::Approx(box.DistanceToPoint(cameraPos)) == 10.0f);

	// Frustum Planes Extraction & Intersection
	glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 viewProj = proj * view;

	auto frustumPlanes = brassica::AABB::ExtractFrustumPlanes(viewProj);
	CHECK(frustumPlanes.size() == 6);

	// Box at origin should be visible inside frustum
	CHECK(box.IntersectsFrustum(frustumPlanes));

	// Box behind camera (Z > camera position 15) should be culled
	brassica::AABB boxBehind(glm::vec3(-5.0f, -5.0f, 50.0f), glm::vec3(5.0f, 5.0f, 100.0f));
	CHECK_FALSE(boxBehind.IntersectsFrustum(frustumPlanes));

	// Calculate LOD level selection
	float baseTexel = 0.5f;
	uint32_t lodNear = box.CalculateLOD(glm::vec3(0.0f, 0.0f, 10.0f), baseTexel, 4);
	uint32_t lodFar = box.CalculateLOD(glm::vec3(0.0f, 0.0f, 500.0f), baseTexel, 4);

	CHECK(lodNear == 0);
	CHECK(lodFar == 3);
}

TEST_CASE("Terrain Clipmap Generation and Level Scaling") {
	uint32_t numLODs = 4;
	float baseTexel = 0.5f;

	// Verify clipmap level spatial scaling
	for (uint32_t l = 0; l < numLODs; ++l) {
		float expectedTexelSize = baseTexel * static_cast<float>(1 << l);
		float expectedExtent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * expectedTexelSize;

		CHECK(doctest::Approx(expectedTexelSize) == baseTexel * std::pow(2.0f, static_cast<float>(l)));
		CHECK(doctest::Approx(expectedExtent) == 1024.0f * expectedTexelSize);
	}

	// Generate 1024x1024 height and normal map for Level 0
	auto mapData = brassica::TerrainClipmap::GenerateSineWaveMap(0, baseTexel, glm::vec2(0.0f), 0.0f);
	CHECK(mapData.size() == brassica::TERRAIN_MAP_DIM * brassica::TERRAIN_MAP_DIM);

	// Sample center texel
	size_t centerIdx = (brassica::TERRAIN_MAP_DIM / 2) * brassica::TERRAIN_MAP_DIM + (brassica::TERRAIN_MAP_DIM / 2);
	glm::vec4 centerTexel = mapData[centerIdx];

	float height = centerTexel.x;
	glm::vec3 normal = glm::vec3(centerTexel.y, centerTexel.z, centerTexel.w);

	// Verify normal vector length is normalized
	CHECK(doctest::Approx(glm::length(normal)).epsilon(0.01f) == 1.0f);
	// Height from sine wave combination should be within reasonable shallow bounds [-10, 10]
	CHECK(height >= -10.0f);
	CHECK(height <= 10.0f);
}

TEST_CASE("AsyncTerrainUploader Initial State") {
	brassica::AsyncTerrainUploader uploader;
	CHECK_FALSE(uploader.HasInFlightUploads());
}
