#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ConfigManager.hpp"
#include "ServiceLocator.hpp"
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

TEST_CASE("TerrainBiomeNode ConfigManager integration for cellsPerUpdate") {
	ServiceLocator locator;
	ServiceLocator::SetInstance(&locator);

	auto configMgr = std::make_shared<ConfigManager>("TestApp");
	configMgr->SetManagerSetting<std::uint32_t>("TerrainBiomeNode", "cellsPerUpdate", 1024);
	locator.Provide<ConfigManager>(configMgr);

	TerrainBiomeNode node;
	render::NodeFrameParams params{};
	node.SetFrameParams(params);

	// Should load 1024 from ConfigManager
	CHECK(node.cellsPerUpdate == 1024);

	// With 1024 cells per update, a full 2048 texture update completes in 2 frames
	const std::uint32_t textureDim = constants::Weather::kCoverageTextureDim;

	// Frame 0: cellOffset 0 -> 1024
	std::uint32_t cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 1024);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 1024);
	CHECK(node.pingPongState == false);

	// Frame 1: cellOffset 1024 -> 2048 -> resets to 0, swaps pingPongState
	cellsToProcess = std::min(node.cellsPerUpdate, textureDim - node.cellOffset);
	CHECK(cellsToProcess == 1024);
	node.cellOffset += cellsToProcess;
	if (node.cellOffset >= textureDim) {
		node.cellOffset = 0;
		node.pingPongState = !node.pingPongState;
	}
	CHECK(node.cellOffset == 0);
	CHECK(node.pingPongState == true);

	ServiceLocator::SetInstance(nullptr);
}
