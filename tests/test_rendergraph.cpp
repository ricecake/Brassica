#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/Node.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "passes/AtmosphereLUTNode.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/GradientNode.hpp"
#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"

using namespace brassica;

TEST_CASE("EngineOptions::FromArgs parses --dot and its variants") {
	const char* argv1[] = {"brassica", "--dot"};
	auto opts1 = EngineOptions::FromArgs(2, const_cast<char**>(argv1));
	CHECK(opts1.dumpDot == true);

	const char* argv2[] = {"brassica", "--dump-dot"};
	auto opts2 = EngineOptions::FromArgs(2, const_cast<char**>(argv2));
	CHECK(opts2.dumpDot == true);

	const char* argv3[] = {"brassica", "--dump-graph-dot"};
	auto opts3 = EngineOptions::FromArgs(2, const_cast<char**>(argv3));
	CHECK(opts3.dumpDot == true);

	const char* argv4[] = {"brassica", "-dot"};
	auto opts4 = EngineOptions::FromArgs(2, const_cast<char**>(argv4));
	CHECK(opts4.dumpDot == true);
}

TEST_CASE("Full Engine frame graph compiles and generates DOT output showing all nodes and temporal abstractions") {
	graph::Graph frameGraph;
	frameGraph.Register<graph::Import<Swapchain>>();
	frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
	frameGraph.Register<graph::PreviousFrame<EngineTemporal>>();
	frameGraph.Register<TransmittanceLUTNode>(TransmittanceLUTNode{});
	frameGraph.Register<MultiScatteringLUTNode>(MultiScatteringLUTNode{});
	frameGraph.Register<GradientNode>(GradientNode{});
	frameGraph.Register<TerrainNode>(TerrainNode{});
	frameGraph.Register<DeferredNode>(DeferredNode{.swapchainFormat = vk::Format::eB8G8R8A8Unorm});
	frameGraph.Register<WaterNode>(WaterNode{.swapchainFormat = vk::Format::eB8G8R8A8Unorm});
	frameGraph.Register<graph::NextFrame<EngineTemporal>>();

	graph::FrameContext ctx{.width = 1280, .height = 720};
	frameGraph.Setup(ctx);

	auto compileResult = frameGraph.Compile();
	REQUIRE(compileResult.has_value());

	std::string dot = graph::ToDot(frameGraph, "Engine Graph Test");

	CHECK(dot.find("digraph") != std::string::npos);
	CHECK(dot.find("PreviousFrame") != std::string::npos);
	CHECK(dot.find("NextFrame") != std::string::npos);
	CHECK(dot.find("Import") != std::string::npos);
	CHECK(dot.find("TransmittanceLUTNode") != std::string::npos);
	CHECK(dot.find("MultiScatteringLUTNode") != std::string::npos);
	CHECK(dot.find("GradientNode") != std::string::npos);
	CHECK(dot.find("TerrainNode") != std::string::npos);
	CHECK(dot.find("DeferredNode") != std::string::npos);
	CHECK(dot.find("WaterNode") != std::string::npos);
	CHECK(dot.find("purple") != std::string::npos);
}

struct NonExistentKey {};

TEST_CASE("Frame graph compilation failure reports error") {
	struct UnmetNode {
		using Resources = graph::Declares<graph::Read<NonExistentKey>>;
		graph::Recipe Setup(const graph::FrameContext&) { return graph::Recipe{}; }
		void Execute(graph::NodeContext&) {}
	};

	graph::Graph frameGraph;
	frameGraph.Register<UnmetNode>();

	graph::FrameContext ctx{.width = 1280, .height = 720};
	frameGraph.Setup(ctx);

	auto compileResult = frameGraph.Compile();
	REQUIRE_FALSE(compileResult.has_value());
	CHECK(compileResult.error().message.find("NonExistentKey") != std::string::npos);
}

// Real GradientNode/TerrainNode/DeferredNode registered into a real graph::Graph -- genuine
// integration coverage the old fg::-based version of this test never had (it exercised
// hand-written MockGradientPass/MockGBufferPass/MockDeferredPass lambdas, never the real pass
// classes). Gated on a real headless device, same pattern as tests/test_headless.cpp/
// test_physical_backend.cpp -- none of these three nodes need a real device to construct
// anymore (no more Pass classes building a real Vulkan pipeline in their constructor), but
// engine.Init() itself still does elsewhere in the engine, so this stays gated the same way.
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

	{
		graph::Graph frameGraph;
		// TerrainClipmapTexture has no producer node -- it's registered once, directly into the
		// registry, at Engine::Init (unexercised here, this test never provisions/executes) --
		// so a plain Import declares it for Compile()'s validation, matching Engine.cpp's real
		// per-frame graph exactly.
		frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
		// pipelineLibrary/shader/terrainAS pointers stay null on all three nodes -- this test's
		// scope is Setup()+Compile() only (see the header comment above), and Execute (the only
		// place those fields are read) is never called here.
		frameGraph.Register<GradientNode>(GradientNode{});
		frameGraph.Register<TerrainNode>(TerrainNode{});
		frameGraph.Register<DeferredNode>(DeferredNode{
			.swapchainFormat = vk::Format::eB8G8R8A8Unorm,
		});

		graph::FrameContext ctx{.width = 1280, .height = 720};
		frameGraph.Setup(ctx);

		auto compileResult = frameGraph.Compile();
		REQUIRE(compileResult.has_value());

		// Real cross-node edges, not just "it compiled": Gradient and the clipmap Import share
		// no dependency with anything, so they land in the first stage (provably independent);
		// Terrain now declares Read<TerrainClipmapTexture> (a real, declared dependency rather
		// than a raw view/sampler smuggled in with no graph edge), so it must land strictly
		// after the Import; Deferred depends on Terrain's outputs too, so it lands later still.
		const auto& schedule = frameGraph.GetSchedule();
		REQUIRE(schedule.stages.size() == 3);
		CHECK(schedule.stages[0].nodes.size() == 2);
		CHECK(schedule.stages[1].nodes.size() == 1);
		CHECK(schedule.stages[2].nodes.size() == 1);
	}

	engine.Cleanup();
}
