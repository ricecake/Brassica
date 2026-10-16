#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/TerrainBiomeNode.hpp"
#include "render/NodeLifecycle.hpp"

using namespace brassica;

TEST_CASE("TerrainBiomeNode lifecycle, recipe setup, and resource realizations") {
	TerrainBiomeNode node;

	render::NodeFrameParams params{};
	params.forceRegeneration = true;
	node.SetFrameParams(params);

	graph::FrameContext ctx{.width = 4096, .height = 4096};
	graph::SetupBuilder builder;
	node.Setup(builder, ctx);

	CHECK(builder.Active() == true);
	CHECK(builder.Domain() == graph::ExecutionDomain::Compute);
	CHECK(builder.Realizations().size() == 7);

	bool foundWeatherBiome = false;
	bool foundPingPong = false;
	bool foundWeatherMapA = false;
	bool foundWeatherMapB = false;
	bool foundBiome = false;
	bool foundBiomeMapA = false;
	bool foundBiomeMapB = false;

	for (const auto& real : builder.Realizations()) {
		CHECK(real.access == graph::AccessKind::ReadWrite);
		if (real.key == graph::IdOf<TerrainWeatherBiomeTexture>()) {
			foundWeatherBiome = true;
		}
		if (real.key == graph::IdOf<TerrainWeatherPingPongTexture>()) {
			foundPingPong = true;
		}
		if (real.key == graph::IdOf<TerrainWeatherMapATexture>()) {
			foundWeatherMapA = true;
		}
		if (real.key == graph::IdOf<TerrainWeatherMapBTexture>()) {
			foundWeatherMapB = true;
		}
		if (real.key == graph::IdOf<TerrainBiomeMainTexture>()) {
			foundBiome = true;
		}
		if (real.key == graph::IdOf<TerrainBiomeMapATexture>()) {
			foundBiomeMapA = true;
		}
		if (real.key == graph::IdOf<TerrainBiomeMapBTexture>()) {
			foundBiomeMapB = true;
		}
	}

	CHECK(foundWeatherBiome);
	CHECK(foundPingPong);
	CHECK(foundWeatherMapA);
	CHECK(foundWeatherMapB);
	CHECK(foundBiome);
	CHECK(foundBiomeMapA);
	CHECK(foundBiomeMapB);

	// Dims now come from each key's own catalog Desc() (ResourceKeys.hpp), not the realization --
	// a builder node's realization carries an empty desc by design (see SetupBuilder.hpp).
	CHECK(TerrainWeatherBiomeTexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainWeatherPingPongTexture{}.Desc(ctx).width == constants::Weather::kSimTextureDim);
	CHECK(TerrainWeatherMapATexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainWeatherMapBTexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainBiomeMainTexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);
	CHECK(TerrainBiomeMapATexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);
	CHECK(TerrainBiomeMapBTexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);
}
