#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "passes/TerrainBiomeNode.hpp"
#include "render/NodeLifecycle.hpp"

using namespace brassica;

TEST_CASE("TerrainBiomeNode lifecycle, recipe setup, and resource realizations") {
	TerrainBiomeNode node;

	render::NodeFrameParams params{};
	params.forceRegeneration = true;
	node.SetFrameParams(params);

	graph::FrameContext ctx{.width = 2048, .height = 2048};
	graph::Recipe recipe = node.Setup(ctx);

	CHECK(recipe.isActive == true);
	CHECK(recipe.domain == graph::ExecutionDomain::Compute);
	CHECK(recipe.realizations.size() == 2);

	bool foundWeatherBiome = false;
	bool foundPingPong = false;

	for (const auto& real : recipe.realizations) {
		if (real.key == graph::IdOf<TerrainWeatherBiomeTexture>()) {
			foundWeatherBiome = true;
			CHECK(real.desc.width == 2048);
			CHECK(real.desc.height == 2048);
		}
		if (real.key == graph::IdOf<TerrainWeatherPingPongTexture>()) {
			foundPingPong = true;
			CHECK(real.desc.width == 2048);
			CHECK(real.desc.height == 2048);
		}
	}

	CHECK(foundWeatherBiome);
	CHECK(foundPingPong);
}
