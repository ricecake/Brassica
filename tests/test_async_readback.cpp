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
