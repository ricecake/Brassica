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
	CHECK(recipe.realizations.size() == 7);

	bool foundWeatherBiome = false;
	bool foundPingPong = false;
	bool foundWeatherMapA = false;
	bool foundWeatherMapB = false;
	bool foundBiome = false;
	bool foundBiomeMapA = false;
	bool foundBiomeMapB = false;

	for (const auto& real : recipe.realizations) {
		if (real.key == graph::IdOf<TerrainWeatherBiomeTexture>()) {
			foundWeatherBiome = true;
			CHECK(real.desc.width == constants::Weather::kCoverageTextureDim);
			CHECK(real.desc.height == constants::Weather::kCoverageTextureDim);
		}
		if (real.key == graph::IdOf<TerrainWeatherPingPongTexture>()) {
			foundPingPong = true;
			CHECK(real.desc.width == constants::Weather::kSimTextureDim);
			CHECK(real.desc.height == constants::Weather::kSimTextureDim);
		}
		if (real.key == graph::IdOf<TerrainWeatherMapATexture>()) {
			foundWeatherMapA = true;
			CHECK(real.desc.width == constants::Weather::kCoverageTextureDim);
			CHECK(real.desc.height == constants::Weather::kCoverageTextureDim);
		}
		if (real.key == graph::IdOf<TerrainWeatherMapBTexture>()) {
			foundWeatherMapB = true;
			CHECK(real.desc.width == constants::Weather::kCoverageTextureDim);
			CHECK(real.desc.height == constants::Weather::kCoverageTextureDim);
		}
		if (real.key == graph::IdOf<TerrainBiomeTexture>()) {
			foundBiome = true;
			CHECK(real.desc.width == constants::Weather::kBiomeTextureDim);
			CHECK(real.desc.height == constants::Weather::kBiomeTextureDim);
		}
		if (real.key == graph::IdOf<TerrainBiomeMapATexture>()) {
			foundBiomeMapA = true;
			CHECK(real.desc.width == constants::Weather::kBiomeTextureDim);
			CHECK(real.desc.height == constants::Weather::kBiomeTextureDim);
		}
		if (real.key == graph::IdOf<TerrainBiomeMapBTexture>()) {
			foundBiomeMapB = true;
			CHECK(real.desc.width == constants::Weather::kBiomeTextureDim);
			CHECK(real.desc.height == constants::Weather::kBiomeTextureDim);
		}
	}

	CHECK(foundWeatherBiome);
	CHECK(foundPingPong);
	CHECK(foundWeatherMapA);
	CHECK(foundWeatherMapB);
	CHECK(foundBiome);
	CHECK(foundBiomeMapA);
	CHECK(foundBiomeMapB);
}
