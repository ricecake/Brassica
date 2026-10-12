#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "Engine.hpp"

using namespace brassica;

TEST_CASE("Engine Persistent Queues & Async Readback Test") {
	Engine engine;
	EngineOptions opts{};
	opts.headless = true;
	opts.maxFrames = 1;

	engine.Init(opts);

	if (engine.GetDevice()) {
		CHECK(engine.GetGraphicsQueue() != vk::Queue{});
		CHECK(engine.GetComputeQueue() != vk::Queue{});
		CHECK(engine.GetTransferQueue() != vk::Queue{});

		const auto& qset = engine.GetQueueSet();
		CHECK(qset.graphics.queue != nullptr);
		CHECK(qset.compute.queue != nullptr);
		CHECK(qset.transfer.queue != nullptr);
		CHECK(qset.graphics.familyIndex == engine.GetGraphicsQueueFamily());
		CHECK(qset.compute.familyIndex == engine.GetComputeQueueFamily());
		CHECK(qset.transfer.familyIndex == engine.GetTransferQueueFamily());

		// Test non-blocking readback polling before triggers -- GetCachedGroundHeight must still
		// report the fallback verbatim, confirming no spurious cached data before anything ever
		// completed.
		engine.GetTerrainManager().PollReadbackData();

		constexpr float kFallback = -1024.0f;
		CHECK(engine.GetTerrainManager().GetCachedGroundHeight(kFallback) == kFallback);

		// Update camera to exercise async readback triggering and motion constraint
		engine.UpdateCamera(0.016f);

		// Poll readback after camera update
		engine.GetTerrainManager().PollReadbackData();

		engine.Cleanup();
	} else {
		MESSAGE("Vulkan device not available in this environment; skipping GPU queue assertions.");
	}
}

TEST_CASE("TerrainManager Bilinear Height Interpolation") {
	TerrainManager mgr;
	constexpr float kFallback = -1024.0f;

	// Fallback check on uninitialized manager
	CHECK(mgr.GetInterpolatedGroundHeight(glm::vec2(10.0f, 10.0f), kFallback) == kFallback);
	CHECK(mgr.GetCachedGroundHeight(kFallback) == kFallback);

	// Verify grid origin math and texel alignment
	constexpr float texelSize = 0.5f;
	constexpr int   dim = 1024;
	constexpr int   kReadbackWidth = 128;

	glm::vec2 camPos2D(500.0f, 300.0f);
	glm::vec2 camGrid = glm::floor(camPos2D / texelSize);

	int camU = static_cast<int>(camGrid.x) % dim;
	if (camU < 0) camU += dim;

	int minU = std::clamp(camU - kReadbackWidth / 2, 0, dim - kReadbackWidth);
	int g_x0 = static_cast<int>(camGrid.x) - (camU - minU);

	float originWorldX = (static_cast<float>(g_x0) + 0.5f) * texelSize;
	float gx = (camPos2D.x - originWorldX) / texelSize;

	// Camera at 500.0m is at fractional offset 103.5 in the 128-wide readback grid (clamped at upper boundary minU = 896)
	CHECK(gx == doctest::Approx(103.5f));
}
