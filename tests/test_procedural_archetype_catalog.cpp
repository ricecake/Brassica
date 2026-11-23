#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "passes/material/MaterialManager.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/ProceduralArchetypeCatalog.hpp"

using namespace brassica;

TEST_CASE("RegisterProceduralArchetypes registers the full fixed catalog cleanly") {
	ModelRegistry   registry;
	MaterialManager materialManager;
	RegisterProceduralArchetypes(registry, materialManager);

	// 6 trees (small/medium/large deciduous + pine/palm/cactus) + 2 bushes + 3 rocks + 3
	// flowers -- the fixed, bounded catalog confirmed with sgreenhusted (not per-instance/
	// per-seed infinite variation).
	REQUIRE(registry.ModelCount() == 14);

	std::uint32_t totalMeshlets = 0;
	for (std::uint32_t m = 0; m < registry.ModelCount(); ++m) {
		std::uint32_t meshletCount = registry.GetMeshletCount(m);
		CHECK(meshletCount > 0); // every variant produced a real, non-empty mesh
		totalMeshlets += meshletCount;
	}
	CHECK(registry.GetTotalMeshletCount() == totalMeshlets);

	// Trees (indices 0-5) and bushes (6-7) are skinned branching structures; rocks (8-10) are
	// jointless (see ManifoldRockAsset.hpp's own comment on why); flowers (11-13) are a short
	// skinned stem capped with a flower-head joint (ManifoldFlowerAsset.hpp).
	for (std::uint32_t m = 0; m < 8; ++m) {
		CHECK(registry.GetJointCount(m) > 1);
		CHECK(registry.GetJointCount(m) <= 24);
	}
	for (std::uint32_t m = 8; m < 11; ++m) {
		CHECK(registry.GetJointCount(m) == 0);
	}
	for (std::uint32_t m = 11; m < 14; ++m) {
		CHECK(registry.GetJointCount(m) > 1);
		CHECK(registry.GetJointCount(m) <= 24);
	}
}
