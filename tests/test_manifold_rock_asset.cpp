#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>

#include "passes/model/ManifoldRockAsset.hpp"
#include "passes/model/ModelRegistry.hpp"

using namespace brassica;

TEST_CASE("BuildProceduralRock produces a real, jointless, non-degenerate mesh") {
	RockGenParams params;
	params.seed = 42;
	ModelAsset rock = BuildProceduralRock(params);

	REQUIRE(!rock.vertices.empty());
	REQUIRE(!rock.meshlets.empty());
	REQUIRE(!rock.meshletVertices.empty());
	REQUIRE(!rock.meshletTrianglesPacked.empty());

	// Jointless-model convention (ModelAsset.hpp's own documented case) -- no skeleton at
	// all, rocks never need wind.
	CHECK(rock.jointCount == 0);
	CHECK(rock.skeleton.empty());
	REQUIRE(rock.palette.size() == 1);
	CHECK(rock.clipTable.empty());

	for (const GltfVertex& v : rock.vertices) {
		CHECK(std::isfinite(v.position.x));
		CHECK(std::isfinite(v.position.y));
		CHECK(std::isfinite(v.position.z));
		float normalLen = glm::length(v.normal);
		CHECK(normalLen > 0.9f);
		CHECK(normalLen < 1.1f);
		// Zero-filled joints/weights -- model_render.mesh's weightSum<=0 fallback renders raw
		// position with no skinning, same convention a non-skinned glTF primitive already uses.
		CHECK((v.weights.x + v.weights.y + v.weights.z + v.weights.w) == doctest::Approx(0.0f));
	}

	CHECK(rock.restBoundsRadius > 0.0f);

	// Two different seeds produce different shapes -- a real check that the cluster-sphere
	// offsets/radii and noise displacement actually depend on the seed, not a hardcoded no-op.
	// Different seeds can legitimately produce different union topology (not just jittered
	// positions on an identical mesh), since the cluster spheres' relative offsets/radii differ
	// -- that's the variety this generator is for, not a bug, so this doesn't assume matching
	// vertex counts.
	RockGenParams params2 = params;
	params2.seed = 1337;
	ModelAsset rock2 = BuildProceduralRock(params2);
	REQUIRE(!rock2.vertices.empty());
	bool anyDifferent = rock2.vertices.size() != rock.vertices.size();
	if (!anyDifferent) {
		for (std::size_t i = 0; i < rock.vertices.size(); ++i) {
			if (glm::length(rock.vertices[i].position - rock2.vertices[i].position) > 1e-4f) {
				anyDifferent = true;
				break;
			}
		}
	}
	CHECK(anyDifferent);

	// Registers cleanly through the unchanged ModelRegistry path.
	ModelRegistry registry;
	std::uint32_t modelIndex = registry.RegisterAsset(rock);
	CHECK(modelIndex == 0);
	CHECK(registry.GetJointCount(modelIndex) == 0);
	CHECK(registry.GetMeshletCount(modelIndex) == rock.meshlets.size());
}
