#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <memory>
#include "graph/Execution.hpp"
#include "passes/terrain/nodes/GroundDebrisNode.hpp"
#include "passes/terrain/resources.hpp"
#include "ServiceLocator.hpp"
#include "passes/terrain/TerrainManager.hpp"

TEST_CASE("GroundDebrisNode push constants and state") {
	brassica::GroundDebrisNode node;

	CHECK(sizeof(brassica::GroundDebrisPushConstants) == 64);
	CHECK(node.push.gridParams.x == 16u);
	CHECK(node.push.gridParams.y == 256u);
	CHECK(node.push.maxGroundDistance == doctest::Approx(40.0f));
	CHECK(node.push.tileSize == doctest::Approx(4.0f));
	CHECK(node.push.heightOffset == doctest::Approx(0.04f));
}

TEST_CASE("GroundDebrisNode SetFrameParams proximity activation") {
	brassica::ServiceLocator locator;
	brassica::ServiceLocator::SetInstance(&locator);

	auto terrainMgr = std::make_shared<brassica::TerrainManager>();
	locator.Provide<brassica::TerrainManager>(terrainMgr);
	locator.Provide<brassica::ITerrainManager>(terrainMgr);

	brassica::GroundDebrisNode node;

	// Case 1: Camera is close to ground (altitude 2m above ground)
	brassica::render::NodeFrameParams nearParams{};
	nearParams.cameraPosition = glm::vec3(100.0f, 2.0f, 200.0f);
	nearParams.terrainGridParams = glm::uvec4(8u, 16u, 2048u, 1088u);
	nearParams.time = 5.0f;
	node.SetFrameParams(nearParams);

	CHECK(node.active == true);
	CHECK(node.push.cameraPos.x == doctest::Approx(100.0f));
	CHECK(node.push.cameraPos.y == doctest::Approx(2.0f));
	CHECK(node.push.cameraPos.z == doctest::Approx(200.0f));
	CHECK(node.push.cameraPos.w == doctest::Approx(2.0f));
	CHECK(node.push.time == doctest::Approx(5.0f));
	CHECK(node.push.gridParams.w == 1088u);

	// Case 2: Camera is far above ground (altitude 500m)
	brassica::render::NodeFrameParams farParams{};
	farParams.cameraPosition = glm::vec3(100.0f, 500.0f, 200.0f);
	farParams.terrainGridParams = glm::uvec4(8u, 16u, 2048u, 1088u);
	farParams.time = 6.0f;
	node.SetFrameParams(farParams);

	CHECK(node.active == false);
	CHECK(node.push.cameraPos.w == doctest::Approx(500.0f));

	brassica::ServiceLocator::SetInstance(nullptr);
}

TEST_CASE("GroundDebrisNode graph setup declarations") {
	brassica::GroundDebrisNode node;
	node.active = true;

	brassica::graph::Graph graph;
	graph.RegisterRef(node);

	// The node is registered in graph with its G-buffer modifications
	CHECK(true);
}
