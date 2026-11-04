#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "model/ModelRegistry.hpp"
#include "model/ProceduralArchetypeCatalog.hpp"

using namespace brassica;

TEST_CASE("RegisterProceduralArchetypes registers the full fixed catalog cleanly") {
	ModelRegistry registry;
	RegisterProceduralArchetypes(registry);

	// 3 trees + 2 bushes + 3 rocks + 3 flowers -- the fixed, bounded catalog confirmed with
	// sgreenhusted (not per-instance/per-seed infinite variation).
	REQUIRE(registry.ModelCount() == 11);

	std::uint32_t totalMeshlets = 0;
	for (std::uint32_t m = 0; m < registry.ModelCount(); ++m) {
		std::uint32_t meshletCount = registry.GetMeshletCount(m);
		CHECK(meshletCount > 0); // every variant produced a real, non-empty mesh
		totalMeshlets += meshletCount;
	}
	CHECK(registry.GetTotalMeshletCount() == totalMeshlets);

	// Trees (indices 0-2) and bushes (3-4) are skinned branching structures; rocks (5-7) are
	// jointless (see ManifoldRockAsset.hpp's own comment on why); flowers (8-10) are a short
	// skinned stem capped with a flower-head joint (ManifoldFlowerAsset.hpp).
	for (std::uint32_t m = 0; m < 5; ++m) {
		CHECK(registry.GetJointCount(m) > 1);
		CHECK(registry.GetJointCount(m) <= 24);
	}
	for (std::uint32_t m = 5; m < 8; ++m) {
		CHECK(registry.GetJointCount(m) == 0);
	}
	for (std::uint32_t m = 8; m < 11; ++m) {
		CHECK(registry.GetJointCount(m) > 1);
		CHECK(registry.GetJointCount(m) <= 24);
	}
}
