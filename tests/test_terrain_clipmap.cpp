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

	glm::vec3 cameraPos(0.0f, 0.0f, 15.0f);
	CHECK(doctest::Approx(box.DistanceToPoint(cameraPos)) == 10.0f);

	glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 viewProj = proj * view;

	auto frustumPlanes = brassica::AABB::ExtractFrustumPlanes(viewProj);
	CHECK(frustumPlanes.size() == 6);

	CHECK(box.IntersectsFrustum(frustumPlanes));

	brassica::AABB boxBehind(glm::vec3(-5.0f, -5.0f, 50.0f), glm::vec3(5.0f, 5.0f, 100.0f));
	CHECK_FALSE(boxBehind.IntersectsFrustum(frustumPlanes));

	float baseTexel = 0.5f;
	uint32_t lodNear = box.CalculateLOD(glm::vec3(0.0f, 0.0f, 10.0f), baseTexel, 4);
	uint32_t lodFar = box.CalculateLOD(glm::vec3(0.0f, 0.0f, 500.0f), baseTexel, 4);

	CHECK(lodNear == 0);
	CHECK(lodFar == 3);
}

TEST_CASE("Quadtree Indirection Map and 8 Level Scaling") {
	uint32_t numLODs = 8;
	float baseTexel = 0.5f;

	for (uint32_t l = 0; l < numLODs; ++l) {
		float expectedTexelSize = baseTexel * static_cast<float>(1 << l);
		float expectedExtent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * expectedTexelSize;

		CHECK(doctest::Approx(expectedTexelSize) == baseTexel * std::pow(2.0f, static_cast<float>(l)));
		CHECK(doctest::Approx(expectedExtent) == 1024.0f * expectedTexelSize);
	}

	float lod6Extent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * (baseTexel * static_cast<float>(1 << 6));
	CHECK(lod6Extent == doctest::Approx(32768.0f));

	float lod7Extent = static_cast<float>(brassica::TERRAIN_MAP_DIM) * (baseTexel * static_cast<float>(1 << 7));
	CHECK(lod7Extent == doctest::Approx(65536.0f));

	auto mapData = brassica::TerrainClipmap::GenerateSineWaveMap(0, baseTexel, glm::vec2(0.0f), 0.0f);
	CHECK(mapData.size() == brassica::TERRAIN_MAP_DIM * brassica::TERRAIN_MAP_DIM);

	size_t centerIdx = (brassica::TERRAIN_MAP_DIM / 2) * brassica::TERRAIN_MAP_DIM + (brassica::TERRAIN_MAP_DIM / 2);
	glm::vec4 centerTexel = mapData[centerIdx];

	float height = centerTexel.x;
	glm::vec3 normal = glm::vec3(centerTexel.y, centerTexel.z, centerTexel.w);

	CHECK(doctest::Approx(glm::length(normal)).epsilon(0.01f) == 1.0f);
	CHECK(height >= -500.0f);
	CHECK(height <= 1500.0f);
}

TEST_CASE("Top Plane Frustum Culling and Terrain Elevation") {
	glm::vec3 cameraPos(0.0f, 15.0f, 30.0f);
	glm::mat4 proj = glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 3000.0f);
	proj[1][1] *= -1.0f;
	glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 viewProj = proj * view;

	auto frustumPlanes = brassica::AABB::ExtractFrustumPlanes(viewProj);

	brassica::AABB elevatedTerrain(glm::vec3(-16.0f, -10.0f, -100.0f), glm::vec3(16.0f, 10.0f, -68.0f));
	CHECK(elevatedTerrain.IntersectsFrustum(frustumPlanes));

	brassica::AABB wayAbove(glm::vec3(-16.0f, 500.0f, -100.0f), glm::vec3(16.0f, 600.0f, -68.0f));
	CHECK_FALSE(wayAbove.IntersectsFrustum(frustumPlanes));
}

TEST_CASE("Quadtree Structure and Node Alignment") {
	brassica::TerrainClipmap clipmap;
	clipmap.Init(vk::Device{}, VK_NULL_HANDLE);

	CHECK(clipmap.GetNumLODs() == 8);
	CHECK(clipmap.GetNumTileSlots() == 64);

	const auto& nodes = clipmap.GetQuadtreeNodes();
	CHECK(nodes.size() == 21845);

	const auto& root = nodes[0];
	CHECK(root.lod == 7);
	CHECK(root.minWorld == glm::vec2(brassica::ROOT_WORLD_MIN_X, brassica::ROOT_WORLD_MIN_Z));
	CHECK(root.maxWorld == glm::vec2(-brassica::ROOT_WORLD_MIN_X, -brassica::ROOT_WORLD_MIN_Z));
}

TEST_CASE("Long Distance Terrain Meshlet Grid Snapping and Coverage") {
	float meshletSize = 32.0f;
	glm::vec3 cameraPos(1234.5f, 20.0f, -567.8f);

	glm::vec2 cameraSnap = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / meshletSize) * meshletSize;

	CHECK(std::fmod(cameraSnap.x, meshletSize) == doctest::Approx(0.0f));
	CHECK(std::fmod(cameraSnap.y, meshletSize) == doctest::Approx(0.0f));

	uint32_t meshletsPerRow = 64;
	float halfExtent = (static_cast<float>(meshletsPerRow) * 0.5f) * meshletSize;

	glm::vec3 gridMin(cameraSnap.x - halfExtent, -50.0f, cameraSnap.y - halfExtent);
	glm::vec3 gridMax(cameraSnap.x + halfExtent, 50.0f, cameraSnap.y + halfExtent);

	brassica::AABB gridAABB(gridMin, gridMax);

	CHECK(gridAABB.DistanceToPoint(cameraPos) == doctest::Approx(0.0f));
}

TEST_CASE("AsyncTerrainUploader Initial State") {
	brassica::AsyncTerrainUploader uploader;
	CHECK_FALSE(uploader.HasInFlightUploads());
}

TEST_CASE("Terrain Attribute Maps Generation") {
	brassica::TerrainClipmap clipmap;
	auto levelData = clipmap.GenerateLevelData(0);

	size_t expectedSize = brassica::TERRAIN_MAP_DIM * brassica::TERRAIN_MAP_DIM;
	CHECK(levelData.heightMap.size() == expectedSize);
	CHECK(levelData.minMaxMap.size() == expectedSize);
	CHECK(levelData.biomeMap.size() == expectedSize);
	CHECK(levelData.visibilityMap.size() == expectedSize);

	size_t sampleIdx = expectedSize / 2;
	glm::vec4 minMaxVal = levelData.minMaxMap[sampleIdx];
	CHECK(minMaxVal.x <= minMaxVal.y);

	glm::vec4 biomeVal = levelData.biomeMap[sampleIdx];
	CHECK(biomeVal.x >= 0.0f);
	CHECK(biomeVal.x <= 1.0f);
	CHECK(biomeVal.y >= 0.0f);

	glm::vec4 visVal = levelData.visibilityMap[sampleIdx];
	CHECK(visVal.x == 1.0f);
}

TEST_CASE("Initial Camera Height and Raytrace AABB Bounds") {
	float baseTexel = 0.5f;
	glm::vec3 camPos(100.0f, 0.0f, -200.0f);

	float sampledH = brassica::TerrainClipmap::SampleTerrain(camPos.x, camPos.z, baseTexel).r;
	float initialCamY = sampledH + 2.0f;

	CHECK(initialCamY == doctest::Approx(sampledH + 2.0f));

	float meshletSize = 32.0f;
	glm::vec3 minB(100.0f, 0.0f, -200.0f);

	float minH = 1e9f;
	float maxH = -1e9f;
	constexpr int numSamples = 5;
	for (int sz = 0; sz < numSamples; ++sz) {
		float tz = static_cast<float>(sz) / static_cast<float>(numSamples - 1);
		float sampleZ = minB.z + tz * meshletSize;
		for (int sx = 0; sx < numSamples; ++sx) {
			float tx = static_cast<float>(sx) / static_cast<float>(numSamples - 1);
			float sampleX = minB.x + tx * meshletSize;
			float h = brassica::TerrainClipmap::SampleTerrain(sampleX, sampleZ, baseTexel).r;
			minH = std::min(minH, h);
			maxH = std::max(maxH, h);
		}
	}
	minB.y = minH - 5.0f;
	float maxBY = maxH + 5.0f;

	CHECK(minB.y < maxBY);
	CHECK(minB.y <= minH);
	CHECK(maxBY >= maxH);
}
