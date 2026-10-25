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
		engine.GetTerrainManager().PollReadbackData(engine.GetDevice());

		constexpr float kFallback = -1024.0f;
		CHECK(engine.GetTerrainManager().GetCachedGroundHeight(kFallback) == kFallback);

		// Update camera to exercise async readback triggering and motion constraint
		engine.UpdateCamera(0.016f);

		// Poll readback after camera update
		engine.GetTerrainManager().PollReadbackData(engine.GetDevice());

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

	// Initialize manager
	mgr.Initialize();

	constexpr float texelSize = constants::Class::Terrain::BaseTexelSize; // 0.5f
	constexpr float dim = static_cast<float>(constants::Class::Terrain::MapDim); // 1088.0f

	// Test PlanReadback origin calculation for positive world position (unclamped vs clamped)
	glm::vec2 camPos2D(500.0f, 500.0f);
	mgr.PlanReadback(camPos2D);

	CHECK(mgr.HasPendingReadback());
	const auto& pending = mgr.GetPendingReadback();

	float gx = (camPos2D.x - pending.originWorld.x) / texelSize;
	float gz = (camPos2D.y - pending.originWorld.y) / texelSize;

	// Camera at 500.0m, 500.0m is centered precisely at offset 63.5 in the 128x128 readback grid
	CHECK(gx == doctest::Approx(63.5f));
	CHECK(gz == doctest::Approx(63.5f));

	// Test PlanReadback origin calculation for clamped boundary case (centerV < 64)
	glm::vec2 camPosClamped(500.0f, 300.0f);
	TerrainManager mgrClamp;
	mgrClamp.Initialize();
	mgrClamp.PlanReadback(camPosClamped);

	CHECK(mgrClamp.HasPendingReadback());
	const auto& pendingClamp = mgrClamp.GetPendingReadback();

	float gxClamp = (camPosClamped.x - pendingClamp.originWorld.x) / texelSize;
	float gzClamp = (camPosClamped.y - pendingClamp.originWorld.y) / texelSize;

	CHECK(gxClamp == doctest::Approx(63.5f));
	CHECK(gzClamp == doctest::Approx(55.5f)); // centerV = 56, minV clamped to 0

	// Test PlanReadback origin calculation for negative world position
	glm::vec2 camPosNegative(-500.0f, -500.0f);
	TerrainManager mgrNeg;
	mgrNeg.Initialize();
	mgrNeg.PlanReadback(camPosNegative);

	CHECK(mgrNeg.HasPendingReadback());
	const auto& pendingNeg = mgrNeg.GetPendingReadback();

	float gxNeg = (camPosNegative.x - pendingNeg.originWorld.x) / texelSize;
	float gzNeg = (camPosNegative.y - pendingNeg.originWorld.y) / texelSize;

	CHECK(gxNeg == doctest::Approx(63.5f));
	CHECK(gzNeg == doctest::Approx(63.5f));
}
