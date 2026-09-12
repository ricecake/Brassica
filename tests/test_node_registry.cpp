#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Engine.hpp"
#include "graph/Graph.hpp"
#include "graph/NodeRegistry.hpp"
#include "passes/DeferredNode.hpp"
#include "passes/GradientNode.hpp"
#include "passes/ResourceKeys.hpp"
#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"

using namespace brassica;

// Dummy node to test custom parameter dispatch
struct DummyTestResource {};

struct DummyNode {
	using Resources = graph::Declares<graph::Create<DummyTestResource>>;

	bool initialized = false;
	bool destroyed = false;
	uint32_t customValue = 0;

	void Init(const graph::NodeInitParams&) {
		initialized = true;
	}

	void SetFrameParams(uint32_t val) {
		customValue = val;
	}

	void Destroy(vk::Device) {
		destroyed = true;
	}

	graph::Recipe Setup(const graph::FrameContext&) {
		return graph::Recipe{.domain = graph::ExecutionDomain::Graphics};
	}

	void Execute(graph::NodeContext&) {}
};

TEST_CASE("NodeRegistry lifecycle and graph creation") {
	graph::NodeRegistry<DummyNode> registry;

	// Check retrieval by type and index
	auto& nodeByType = registry.GetNode<DummyNode>();
	auto& nodeByIndex = registry.GetNode<0>();
	CHECK(&nodeByType == &nodeByIndex);
	CHECK_FALSE(nodeByType.initialized);

	graph::NodeInitParams initParams{};
	registry.Init(initParams);
	CHECK(nodeByType.initialized);

	graph::NodeFrameParams frameParams{};
	registry.UpdateFrameParams(frameParams);

	graph::Graph graph = registry.CreateGraph();
	CHECK(graph.Nodes().size() == 1);

	registry.Destroy(vk::Device{});
	CHECK(nodeByType.destroyed);
}

TEST_CASE("DefaultNodeRegistry integration with Engine") {
	brassica::Engine        engine;
	brassica::EngineOptions opts;
	opts.headless = true;
	engine.Init(opts);

	if (!engine.GetDevice()) {
		MESSAGE("Vulkan physical device not available in this environment; skipping GPU execution.");
		return;
	}

	// Verify nodes are accessible via Engine's GetNode<T>() and GetNodeRegistry()
	auto& registry = engine.GetNodeRegistry();
	auto& gradientNode = engine.GetNode<GradientNode>();
	auto& terrainNode = engine.GetNode<TerrainNode>();
	auto& deferredNode = engine.GetNode<DeferredNode>();
	auto& waterNode = engine.GetNode<WaterNode>();

	CHECK(&gradientNode == &registry.GetNode<GradientNode>());
	CHECK(&terrainNode == &registry.GetNode<TerrainNode>());
	CHECK(&deferredNode == &registry.GetNode<DeferredNode>());
	CHECK(&waterNode == &registry.GetNode<WaterNode>());

	graph::Graph frameGraph;
	frameGraph.Register<graph::Import<TerrainClipmapTexture>>();
	registry.PopulateGraph<GradientNode, TerrainNode, DeferredNode, WaterNode>(frameGraph);

	graph::FrameContext ctx{.width = 1280, .height = 720};
	frameGraph.Setup(ctx);

	auto compileResult = frameGraph.Compile();
	REQUIRE(compileResult.has_value());

	const auto& schedule = frameGraph.GetSchedule();
	REQUIRE(schedule.stages.size() == 3);

	engine.Cleanup();
}
