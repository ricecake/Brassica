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
	mgr.SetGlobalProperties(props);

	brassica::GlobalGrassProperties updatedProps = mgr.GetGlobalProperties();
	CHECK(updatedProps.densityMultiplier == doctest::Approx(2.5f));
	CHECK(updatedProps.lengthMultiplier == doctest::Approx(1.2f));

	brassica::GrassProperties biome0 = mgr.GetBiomeProperties(0);
	CHECK(biome0.enabled == 1u);
	CHECK(biome0.height > 0.0f);
}

TEST_CASE("FoliageNode registration and phase") {
	CHECK(brassica::FoliageNode::kPhase == brassica::SubPhase::GBuffer);
}
