#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/DeferredPass.hpp"
#include "passes/GradientPass.hpp"
#include "passes/TerrainPass.hpp"

using namespace brassica;

// Real GradientNode/TerrainNode/DeferredNode registered into a real graph::Graph -- genuine
// integration coverage the old fg::-based version of this test never had (it exercised
// hand-written MockGradientPass/MockGBufferPass/MockDeferredPass lambdas, never the real pass
// classes). Gated on a real headless device, same pattern as
// tests/test_headless.cpp/test_physical_backend.cpp, since TerrainPass/DeferredPass build real
// Vulkan pipelines (and TerrainPass compiles real mesh/task shaders) in their constructors.
//
// Only Setup()+Compile() are exercised here, matching the old test's own scope (it never called
// fg.execute() either) -- this proves the real production nodes compose into a renderable,
// correctly-scheduled graph; PhysicalExecutionBackend's actual Provision/barrier/render path has
// its own dedicated coverage in tests/test_physical_backend.cpp.
TEST_CASE("Real GradientNode/TerrainNode/DeferredNode compose into a renderable, correctly-staged graph") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	vk::Device   device = engine.GetDevice();
	vk::Instance instance = engine.GetInstance();

	{
		// A minimal, valid (if trivial) stand-in for the real global descriptor set 0 layout --
		// TerrainPass/DeferredPass's pipeline layouts reference it by slot, so it must be a real
		// vk::DescriptorSetLayout, even though nothing in this test binds an actual descriptor
		// set to it. Scoped in a nested block, along with every other device-dependent object
		// below, so all of it is destroyed before engine.Cleanup() tears down the device --
		// Vulkan handles held by locals must not outlive the device that owns them.
		vk::DescriptorSetLayout globalSet0Layout =
			device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{});

		GradientPass gradientPass(device, vk::Format::eR16G16B16A16Sfloat);
		TerrainPass  terrainPass(instance, device, globalSet0Layout);
		DeferredPass deferredPass(device, globalSet0Layout, vk::Format::eB8G8R8A8Unorm);

		graph::PhysicalResourceRegistry registry(device, engine.GetAllocator());

		graph::Graph frameGraph;
		frameGraph.Register<GradientNode>(GradientNode{.pass = &gradientPass, .extent = {1280, 720}});
		frameGraph.Register<TerrainNode>(
			TerrainNode{
				.pass = &terrainPass,
				.extent = {1280, 720},
				.globalDescriptorSet = nullptr,
				.pushConstants = {},
			}
		);
		frameGraph.Register<DeferredNode>(
			DeferredNode{
				.pass = &deferredPass,
				.registry = &registry,
				.extent = {1280, 720},
				.swapchainFormat = vk::Format::eB8G8R8A8Unorm,
				.globalDescriptorSet = nullptr,
				.activeFrame = 0,
				.clipmapImageView = nullptr,
				.clipmapSampler = nullptr,
				.pushConstants = {},
			}
		);

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		// Real cross-node edges, not just "it compiled": Gradient and Terrain share no
		// dependency between them, so they must land in the same stage (provably independent);
		// Deferred depends on both of their outputs (G-buffer, gradient background, TLAS), so it
		// must land strictly later.
		const auto& schedule = frameGraph.GetSchedule();
		REQUIRE(schedule.stages.size() == 2);
		CHECK(schedule.stages[0].nodes.size() == 2);
		CHECK(schedule.stages[1].nodes.size() == 1);

		device.destroyDescriptorSetLayout(globalSet0Layout);
	}

	engine.Cleanup();
}
