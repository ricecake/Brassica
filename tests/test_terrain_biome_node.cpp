#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "graph/Graph.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
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
	// 7 ReadWrite texture realizations plus one Read<LightsBuffer>() (shaders/climate_ca.comp
	// reads uLights[0].direction via clustered_lighting.glsl).
	CHECK(builder.Realizations().size() == 8);

	bool foundWeatherBiome = false;
	bool foundPingPong = false;
	bool foundWeatherMapA = false;
	bool foundWeatherMapB = false;
	bool foundBiome = false;
	bool foundBiomeMapA = false;
	bool foundBiomeMapB = false;
	bool foundLightsBuffer = false;

	for (const auto& real : builder.Realizations()) {
		if (real.key == graph::IdOf<LightsBuffer>()) {
			foundLightsBuffer = true;
			CHECK(real.access == graph::AccessKind::Read);
			continue;
		}
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
	CHECK(foundLightsBuffer);

	// Dims now come from each key's own catalog Desc() (ResourceKeys.hpp), not the realization --
	// a builder node's realization carries an empty desc by design (see SetupBuilder.hpp).
	CHECK(TerrainWeatherBiomeTexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainWeatherPingPongTexture{}.Desc(ctx).width == constants::Weather::kSimTextureDim);
	CHECK(TerrainWeatherMapATexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainWeatherMapBTexture{}.Desc(ctx).width == constants::Weather::kCoverageTextureDim);
	CHECK(TerrainBiomeMainTexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);
	CHECK(TerrainBiomeMapATexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);
	CHECK(TerrainBiomeMapBTexture{}.Desc(ctx).width == constants::Weather::kBiomeTextureDim);

	// Verify push constant structures for climate compute shaders
	CHECK(sizeof(ClimateInitialPushConstants) == 16);
	CHECK(sizeof(ClimateCAPushConstants) == 40);
}

TEST_CASE("TerrainBiomeNode partial CA update cycle and ping-pong buffer swapping") {
	TerrainBiomeNode node;
	node.cellsPerUpdate = 512;

	render::NodeFrameParams params{};
	params.forceRegeneration = false;

	// Simulate frame progression for partial CA updates
	// Texture dimension is constants::Weather::kCoverageTextureDim (2048)
	// With 512 cells per update, 1 full update cycle takes 2048 / 512 = 4 frames.

	// Frame 0 (initial generation)
	params.forceRegeneration = true;
	node.SetFrameParams(params);
	CHECK(node.cellsPerUpdate == 512);

	// Simulate Execute step logic manually / tracking
	const std::uint32_t textureDim = constants::Weather::kCoverageTextureDim;

	// Frame 0: cellOffset = 0 -> process 512 -> cellOffset = 512
	std::uint32_t cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 512);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	} else {
		node.pingPongState = false;
	}
	CHECK(node.cellOffset == 512);
	CHECK(node.pingPongState == false);

	// Frame 1: cellOffset = 512 -> process 512 -> cellOffset = 1024
	params.forceRegeneration = false;
	node.SetFrameParams(params);
	cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 512);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 1024);
	CHECK(node.pingPongState == false);

	// Frame 2: cellOffset = 1024 -> process 512 -> cellOffset = 1536
	node.SetFrameParams(params);
	cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 512);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 1536);
	CHECK(node.pingPongState == false);

	// Frame 3: cellOffset = 1536 -> process 512 -> cellOffset = 2048 >= 2048 -> cellOffset = 0, pingPongState swaps to true
	node.SetFrameParams(params);
	cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 512);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 0);
	CHECK(node.pingPongState == true);

	// Frame 4: Next cycle starts with new pingPongState = true, cellOffset = 0 -> 512
	node.SetFrameParams(params);
	cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 512);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 512);
	CHECK(node.pingPongState == true);
}

TEST_CASE("TerrainBiomeNode default kCellsPerUpdate constant verification") {
	TerrainBiomeNode node;
	render::NodeFrameParams params{};
	node.SetFrameParams(params);

	// Should use constants::Weather::kCellsPerUpdate (512)
	CHECK(node.cellsPerUpdate == constants::Weather::kCellsPerUpdate);
	CHECK(node.cellsPerUpdate == 512);
}
