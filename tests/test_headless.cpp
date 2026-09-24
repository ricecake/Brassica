#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

TEST_CASE("Engine Headless Render Initialization and Execution") {
	brassica::EngineOptions options;
	options.headless = true;
	options.maxFrames = 10;

	brassica::Engine engine;
	engine.Init(options);

	CHECK(engine.GetOptions().headless == true);

	if (engine.GetDevice()) {
		CHECK(static_cast<bool>(engine.GetDevice()));
		CHECK(engine.GetAllocator() != VK_NULL_HANDLE);

		auto props2 = engine.GetPhysicalDevice().getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
		if (props2.get<vk::PhysicalDeviceDriverProperties>().driverID == vk::DriverId::eMesaLlvmpipe) {
			MESSAGE("Mesa LLVMpipe software driver detected; skipping Mesh Shader GPU dispatches in LLVMpipe JIT.");
		} else {
			engine.Run();
		}
		engine.Cleanup();

		CHECK(engine.GetValidationErrorCount() == 0);
		CHECK(engine.GetValidationWarningCount() == 0);
	} else {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
	}
}
