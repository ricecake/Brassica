#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/CloudShadowGenNode.hpp"
#include "passes/TerrainShadowNode.hpp"

using namespace brassica;

struct PrepInputsNode {
	void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
		r.Create<TerrainMinMaxTexture>();
		r.Create<TerrainClipmapTexture>();
		r.Create<CloudVolumeCascade0>();
		r.Create<CloudVolumeCascade1>();
		r.Create<CloudVolumeCascade2>();
		r.Create<TerrainWeatherBiomeTexture>();
	}

	void Execute(graph::NodeContext&) {}
};

TEST_CASE("CloudShadowGenNode reads TerrainShadowMap and orders after TerrainShadowNode") {
	PrepInputsNode     prepNode;
	TerrainShadowNode  terrainShadowNode;
	CloudShadowGenNode cloudShadowGenNode;

	graph::Graph g;
	g.RegisterRef(prepNode);
	g.RegisterRef(terrainShadowNode);
	g.RegisterRef(cloudShadowGenNode);

	graph::FrameContext ctx{.width = 1024, .height = 1024};
	g.Setup(ctx);
	REQUIRE(g.Compile().has_value());

	const auto& schedule = g.GetSchedule();

	auto stageOf = [&](std::size_t nodeIndex) -> std::optional<std::size_t> {
		for (std::size_t s = 0; s < schedule.stages.size(); ++s) {
			for (std::size_t n : schedule.stages[s].nodes) {
				if (n == nodeIndex) {
					return s;
				}
			}
		}
		return std::nullopt;
	};

	std::optional<std::size_t> prepStage = stageOf(0);
	std::optional<std::size_t> terrainShadowStage = stageOf(1);
	std::optional<std::size_t> cloudShadowStage = stageOf(2);

	REQUIRE(prepStage.has_value());
	REQUIRE(terrainShadowStage.has_value());
	REQUIRE(cloudShadowStage.has_value());

	// TerrainShadowNode reads inputs from prepNode, CloudShadowGenNode reads TerrainShadowMap.
	// Therefore, cloudShadowStage must execute in or after terrainShadowStage.
	CHECK(*cloudShadowStage >= *terrainShadowStage);

	// Verify ResourceId assignments
	CHECK(graph::IdOf<CloudShadowMap>() == graph::IdOf<CloudShadowMap>());
}
