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

		// Test non-blocking readback polling before triggers
		engine.PollReadbackData();

		std::vector<glm::vec4> readbackData;
		uint32_t rw = 0, rh = 0;
		bool hasData = engine.GetLatestReadbackData(readbackData, rw, rh);
		CHECK_FALSE(hasData);

		// Update camera to exercise async readback triggering and motion constraint
		engine.UpdateCamera(0.016f);

		// Poll readback after camera update
		engine.PollReadbackData();

		engine.Cleanup();
	} else {
		MESSAGE("Vulkan device not available in this environment; skipping GPU queue assertions.");
	}
}
