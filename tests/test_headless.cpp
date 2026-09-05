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
	CHECK(static_cast<bool>(engine.GetDevice()));
	CHECK(engine.GetAllocator() != VK_NULL_HANDLE);

	engine.Run();
	engine.Cleanup();
}
