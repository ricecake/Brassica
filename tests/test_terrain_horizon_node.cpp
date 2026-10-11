#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <optional>

#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "passes/TerrainGenNode.hpp"
#include "passes/TerrainHorizonNode.hpp"

using namespace brassica;

TEST_CASE("TerrainHorizonNode graph scheduling and barrier synthesis") {
	TerrainGenNode     genNode;
	TerrainHorizonNode horizonNode;

	graph::Graph g;
	g.RegisterRef(genNode);
	g.RegisterRef(horizonNode);

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
	std::optional<std::size_t> horizonStage = stageOf(1);
	REQUIRE(genStage.has_value());
	REQUIRE(horizonStage.has_value());

	// TerrainHorizonNode consumes TerrainClipmapTexture produced by TerrainGenNode
	CHECK(*horizonStage >= *genStage);

	const graph::ResourceId clipmapId = graph::IdOf<TerrainClipmapTexture>();
	bool                    foundBarrier = false;
	for (std::size_t s = *genStage; s <= *horizonStage; ++s) {
		for (const auto& b : schedule.stages[s].preBarriers.Items()) {
			if (b.resource == clipmapId) {
				foundBarrier = true;
			}
		}
	}
	CHECK(foundBarrier);
}
