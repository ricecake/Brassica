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

	graph::FrameContext ctx{.width = 4096, .height = 4096};
	graph::Recipe recipe = node.Setup(ctx);

	CHECK(recipe.isActive == true);
	CHECK(recipe.domain == graph::ExecutionDomain::Compute);
	CHECK(recipe.realizations.size() == 4);

	bool foundWeatherBiome = false;
	bool foundPingPong = false;
	bool foundMapA = false;
	bool foundMapB = false;

	for (const auto& real : recipe.realizations) {
		if (real.key == graph::IdOf<TerrainWeatherBiomeTexture>()) {
			foundWeatherBiome = true;
			CHECK(real.desc.width == 4096);
			CHECK(real.desc.height == 4096);
		}
		if (real.key == graph::IdOf<TerrainWeatherPingPongTexture>()) {
			foundPingPong = true;
			CHECK(real.desc.width == 4096);
			CHECK(real.desc.height == 4096);
		}
		if (real.key == graph::IdOf<TerrainWeatherMapATexture>()) {
			foundMapA = true;
			CHECK(real.desc.width == 4096);
			CHECK(real.desc.height == 4096);
		}
		if (real.key == graph::IdOf<TerrainWeatherMapBTexture>()) {
			foundMapB = true;
			CHECK(real.desc.width == 4096);
			CHECK(real.desc.height == 4096);
		}
	}

	CHECK(foundWeatherBiome);
	CHECK(foundPingPong);
	CHECK(foundMapA);
	CHECK(foundMapB);
}
