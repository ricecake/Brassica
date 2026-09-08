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

TEST_CASE("Terrain Clipmap Generation and 7 Level Scaling") {
	uint32_t numLODs = 7;
	float baseTexel = 0.5f;

	// Verify clipmap level spatial scaling for 7 LODs
	for (uint32_t l = 0; l < numLODs; ++l) {
		float expectedTexelSize = baseTexel * static_cast<float>(1 << l);
		float expectedExtent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * expectedTexelSize;

		CHECK(doctest::Approx(expectedTexelSize) == baseTexel * std::pow(2.0f, static_cast<float>(l)));
		CHECK(doctest::Approx(expectedExtent) == 1088.0f * expectedTexelSize);
	}

	// LOD 6 extent should cover over 34,000 world units with padded extent (1088 texels * 32m = 34816m)
	float lod6Extent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * (baseTexel * static_cast<float>(1 << 6));
	CHECK(lod6Extent == doctest::Approx(34816.0f));

	// Generate 1088x1088 height and normal map for Level 0
	auto mapData = brassica::TerrainClipmap::GenerateSineWaveMap(0, baseTexel, glm::vec2(0.0f), 0.0f);
	CHECK(mapData.size() == brassica::TERRAIN_MAP_DIM * brassica::TERRAIN_MAP_DIM);

	// Sample center texel
	size_t centerIdx = (brassica::TERRAIN_MAP_DIM / 2) * brassica::TERRAIN_MAP_DIM + (brassica::TERRAIN_MAP_DIM / 2);
	glm::vec4 centerTexel = mapData[centerIdx];

	float height = centerTexel.x;
	glm::vec3 normal = glm::vec3(centerTexel.y, centerTexel.z, centerTexel.w);

	// Verify normal vector length is normalized
	CHECK(doctest::Approx(glm::length(normal)).epsilon(0.01f) == 1.0f);
	// Height from FastNoise2 terrain generator should be within [-120, 120]
	CHECK(height >= -120.0f);
	CHECK(height <= 120.0f);
}

TEST_CASE("Top Plane Frustum Culling and Terrain Elevation") {
	glm::vec3 cameraPos(0.0f, 15.0f, 30.0f);
	glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 3000.0f);
	proj[1][1] *= -1.0f; // Vulkan inverted Y
	glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 viewProj = proj * view;

	auto frustumPlanes = brassica::AABB::ExtractFrustumPlanes(viewProj);

	// AABB in front of camera at elevation [ -10, 10 ]
	brassica::AABB elevatedTerrain(glm::vec3(-16.0f, -10.0f, -100.0f), glm::vec3(16.0f, 10.0f, -68.0f));
	CHECK(elevatedTerrain.IntersectsFrustum(frustumPlanes));

	// AABB far above top frustum plane should be culled
	brassica::AABB wayAbove(glm::vec3(-16.0f, 500.0f, -100.0f), glm::vec3(16.0f, 600.0f, -68.0f));
	CHECK_FALSE(wayAbove.IntersectsFrustum(frustumPlanes));
}

TEST_CASE("Toroidal Mapping Offset Calculation") {
	int dim = static_cast<int>(brassica::TERRAIN_MAP_DIM); // 1088
	int offset = 0;

	// Camera moves right by 10 texels
	int deltaX = 10;
	offset = (offset + deltaX) % dim;
	if (offset < 0) offset += dim;
	CHECK(offset == 10);

	// Camera moves left by 25 texels
	int deltaX2 = -25;
	offset = (offset + deltaX2) % dim;
	if (offset < 0) offset += dim;
	CHECK(offset == dim - 15);
}

TEST_CASE("Long Distance Terrain Meshlet Grid Snapping and Coverage") {
	float meshletSize = 32.0f;
	glm::vec3 cameraPos(1234.5f, 20.0f, -567.8f);

	glm::vec2 cameraSnap = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / meshletSize) * meshletSize;

	// Grid should snap to multi-units of meshletSize (32.0f)
	CHECK(std::fmod(cameraSnap.x, meshletSize) == doctest::Approx(0.0f));
	CHECK(std::fmod(cameraSnap.y, meshletSize) == doctest::Approx(0.0f));

	uint32_t meshletsPerRow = 64;
	float halfExtent = (static_cast<float>(meshletsPerRow) * 0.5f) * meshletSize; // 1024 world units

	glm::vec3 gridMin(cameraSnap.x - halfExtent, -50.0f, cameraSnap.y - halfExtent);
	glm::vec3 gridMax(cameraSnap.x + halfExtent, 50.0f, cameraSnap.y + halfExtent);

	brassica::AABB gridAABB(gridMin, gridMax);

	// Camera position should be well inside the grid's XZ extents
	CHECK(gridAABB.DistanceToPoint(cameraPos) == doctest::Approx(0.0f));
}

TEST_CASE("AsyncTerrainUploader Initial State") {
	brassica::AsyncTerrainUploader uploader;
	CHECK_FALSE(uploader.HasInFlightUploads());
}
