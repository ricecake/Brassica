#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainGlobalNode.hpp"
#include "passes/TerrainMinMaxDownsampleNode.hpp"
#include "passes/TerrainShadowNode.hpp"

using namespace brassica;

TEST_CASE("TerrainShadowNode registers, compiles graph dependencies, and orders after TerrainMinMaxDownsampleNode") {
	TerrainGenNode              genNode;
	TerrainGlobalNode           globalNode;
	TerrainMinMaxDownsampleNode downsampleNode;
	TerrainShadowNode           shadowNode;

	graph::Graph g;
	g.RegisterRef(genNode);
	g.RegisterRef(globalNode);
	g.RegisterRef(downsampleNode);
	g.RegisterRef(shadowNode);

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

	std::optional<std::size_t> genStage = stageOf(0);
	std::optional<std::size_t> downsampleStage = stageOf(1);
	std::optional<std::size_t> shadowStage = stageOf(2);

	REQUIRE(genStage.has_value());
	REQUIRE(downsampleStage.has_value());
	REQUIRE(shadowStage.has_value());

	// TerrainShadowNode reads TerrainMinMaxTexture, which downsampleNode Modifies.
	// Therefore, shadowStage must execute after downsampleStage.
	CHECK(*shadowStage >= *downsampleStage);

	// Verify ResourceId of TerrainShadowMap is assigned
	const graph::ResourceId shadowMapId = graph::IdOf<TerrainShadowMap>();
	CHECK(shadowMapId == graph::IdOf<TerrainShadowMap>());
}
