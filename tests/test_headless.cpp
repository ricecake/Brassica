#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"

TEST_CASE("EngineOptions Mock ICD and Driver Choice Defaults") {
	SUBCASE("Default non-headless mode uses real driver by default") {
		const char* argv[] = {"app"};
		auto opts = brassica::EngineOptions::FromArgs(1, const_cast<char**>(argv));
		CHECK(opts.headless == false);
		CHECK(opts.useMockICD == false);
	}

	SUBCASE("Headless mode enables mock ICD by default") {
		const char* argv[] = {"app", "--headless"};
		auto opts = brassica::EngineOptions::FromArgs(2, const_cast<char**>(argv));
		CHECK(opts.headless == true);
		CHECK(opts.useMockICD == true);
	}

	SUBCASE("Headless mode with --use-real-driver overrides mock ICD") {
		const char* argv[] = {"app", "--headless", "--use-real-driver"};
		auto opts = brassica::EngineOptions::FromArgs(3, const_cast<char**>(argv));
		CHECK(opts.headless == true);
		CHECK(opts.useMockICD == false);
		CHECK(opts.forceRealDriver == true);
	}

	SUBCASE("Non-headless mode with --use-mock-icd enables mock ICD") {
		const char* argv[] = {"app", "--use-mock-icd"};
		auto opts = brassica::EngineOptions::FromArgs(2, const_cast<char**>(argv));
		CHECK(opts.headless == false);
		CHECK(opts.useMockICD == true);
	}
}

TEST_CASE("Engine Headless Render Initialization and Execution") {
	brassica::EngineOptions options;
	options.headless = true;
	options.useMockICD = true;
	options.maxFrames = 10;

	brassica::Engine engine;
	engine.Init(options);

	CHECK(engine.GetOptions().headless == true);
	CHECK(engine.GetOptions().useMockICD == true);

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
