#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "foliage/FoliageManager.hpp"
#include "passes/FoliageNode.hpp"
#include "ServiceLocator.hpp"

TEST_CASE("FoliageManager state and global properties") {
	brassica::FoliageManager mgr;
	mgr.Initialize();

	CHECK(mgr.IsEnabled() == true);

	mgr.SetEnabled(false);
	CHECK(mgr.IsEnabled() == false);

	brassica::GlobalGrassProperties props = mgr.GetGlobalProperties();
	props.densityMultiplier = 2.5f;
	props.lengthMultiplier = 1.2f;
	props.rockRatio = 0.12f;
	props.seaweedRatio = 0.18f;
	props.bushRatio = 0.25f;
	props.treeRatio = 0.35f;
	props.lodBaseRange = 500.0f;
	props.maxLODs = 10u;
	props.tilesPerRow = 24u;
	mgr.SetGlobalProperties(props);

	brassica::GlobalGrassProperties updatedProps = mgr.GetGlobalProperties();
	CHECK(updatedProps.densityMultiplier == doctest::Approx(2.5f));
	CHECK(updatedProps.lengthMultiplier == doctest::Approx(1.2f));
	CHECK(updatedProps.rockRatio == doctest::Approx(0.12f));
	CHECK(updatedProps.seaweedRatio == doctest::Approx(0.18f));
	CHECK(updatedProps.bushRatio == doctest::Approx(0.25f));
	CHECK(updatedProps.treeRatio == doctest::Approx(0.35f));
	CHECK(updatedProps.lodBaseRange == doctest::Approx(500.0f));
	CHECK(updatedProps.maxLODs == 10u);
	CHECK(updatedProps.tilesPerRow == 24u);

	// All 10 biomes are enabled by default -- a biome's weather classification can legitimately
	// drift into any of them over a long session (shaders/helpers/whittaker.glsl), and one left
	// disabled by default meant foliage could silently vanish wherever that happened.
	for (std::uint32_t i = 0; i < brassica::kFoliageBiomeCount; ++i) {
		brassica::GrassProperties biome = mgr.GetBiomeProperties(i);
		CHECK(biome.enabled == 1u);
	}
	brassica::GrassProperties biome1 = mgr.GetBiomeProperties(1);
	CHECK(biome1.height > 0.0f);
}

TEST_CASE("FoliageNode registration and push constants sync") {
	CHECK(brassica::FoliageNode::kPhase == brassica::SubPhase::GBuffer);

	brassica::ServiceLocator locator;
	brassica::ServiceLocator::SetInstance(&locator);

	auto foliageMgr = std::make_shared<brassica::FoliageManager>();
	foliageMgr->Initialize();
	locator.Provide<brassica::IFoliageManager>(foliageMgr);

	brassica::GlobalGrassProperties props = foliageMgr->GetGlobalProperties();
	props.lodBaseRange = 800.0f;
	props.rockRatio = 0.3f;
	props.seaweedRatio = 0.2f;
	props.bushRatio = 0.4f;
	props.treeRatio = 0.5f;
	props.maxLODs = 12u;
	props.tilesPerRow = 16u;
	foliageMgr->SetGlobalProperties(props);

	brassica::FoliageNode node;
	brassica::render::NodeFrameParams frameParams{.time = 12.34f};
	node.SetFrameParams(frameParams);

	CHECK(node.push.windTime == doctest::Approx(12.34f));
	CHECK(node.push.lodBaseRange == doctest::Approx(800.0f));
	CHECK(node.push.rockRatio == doctest::Approx(0.3f));
	CHECK(node.push.seaweedRatio == doctest::Approx(0.2f));
	CHECK(node.push.bushRatio == doctest::Approx(0.4f));
	CHECK(node.push.treeRatio == doctest::Approx(0.5f));
	CHECK(node.push.gridParams.x == 12u);
	CHECK(node.push.gridParams.y == 16u);
	CHECK(node.push.gridParams.z == 12u * 16u * 16u);
	CHECK(node.push.hizIndex == 0u);

	brassica::ServiceLocator::SetInstance(nullptr);
}
