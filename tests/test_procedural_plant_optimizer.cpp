#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "passes/model/ProceduralPlantOptimizer.hpp"

using namespace brassica;

TEST_CASE("ConsolidateTubes merges a perfectly collinear run into one tube, no ControlPoint") {
	PlantIR ir;
	int     a = ir.AddTube({0, 0, 0}, {0, 1, 0}, 0.2f, 0.18f, 1.0f);
	int     b = ir.AddTube({0, 1, 0}, {0, 2, 0}, 0.18f, 0.16f, 1.0f, a);
	ir.AddTube({0, 2, 0}, {0, 3, 0}, 0.16f, 0.14f, 1.0f, b);

	ProceduralPlantOptimizer::Optimize(ir);

	// All three tubes are collinear (same +Y direction) -- they should collapse into a single
	// tube spanning the full run, with no surviving ControlPoint or Hub (no branching at all).
	int tubeCount = 0;
	for (const PlantElement& e : ir.elements) {
		if (e.type == PlantElementType::Tube) {
			++tubeCount;
		}
		CHECK(e.type != PlantElementType::ControlPoint);
		CHECK(e.type != PlantElementType::Hub);
	}
	REQUIRE(tubeCount == 1);

	const PlantElement& merged = ir.elements[0];
	CHECK(merged.position == glm::vec3(0, 0, 0));
	CHECK(merged.endPosition == glm::vec3(0, 3, 0));
	CHECK(merged.radius == doctest::Approx(0.2f));
	CHECK(merged.endRadius == doctest::Approx(0.14f));
}

TEST_CASE("ConsolidateTubes preserves a real bend as a ControlPoint instead of straightening it") {
	PlantIR ir;
	int     a = ir.AddTube({0, 0, 0}, {0, 1, 0}, 0.2f, 0.18f, 1.0f);
	// A sharp bend, not a straight continuation.
	ir.AddTube({0, 1, 0}, {1, 1, 0}, 0.18f, 0.16f, 1.0f, a);

	ProceduralPlantOptimizer::Optimize(ir);

	int tubeCount = 0;
	int controlPointCount = 0;
	for (const PlantElement& e : ir.elements) {
		if (e.type == PlantElementType::Tube) {
			++tubeCount;
		} else if (e.type == PlantElementType::ControlPoint) {
			++controlPointCount;
			CHECK(e.position == glm::vec3(0, 1, 0));
		}
	}
	// The two tubes stay separate (a real bend), bridged by one ControlPoint at the join.
	CHECK(tubeCount == 2);
	CHECK(controlPointCount == 1);
}

TEST_CASE("HandleJunctions inserts a flare Hub at a real branch point") {
	PlantIR ir;
	int     trunk = ir.AddTube({0, 0, 0}, {0, 1, 0}, 0.3f, 0.25f, 1.0f);
	ir.AddTube({0, 1, 0}, {1, 2, 0}, 0.2f, 0.15f, 0.5f, trunk);
	ir.AddTube({0, 1, 0}, {-1, 2, 0}, 0.2f, 0.15f, 0.5f, trunk);

	ProceduralPlantOptimizer::Optimize(ir);

	int hubCount = 0;
	for (const PlantElement& e : ir.elements) {
		if (e.type == PlantElementType::Hub) {
			++hubCount;
			CHECK(e.position == glm::vec3(0, 1, 0));
			// Flare: a little larger than the abutting tube's own end radius (0.25), not just a
			// bare pass-through.
			CHECK(e.radius > 0.25f);
			CHECK(e.children.size() == 2);
		}
	}
	REQUIRE(hubCount == 1);

	// The trunk's own single child is now the hub, not the two branches directly.
	const PlantElement& trunkElem = ir.elements[trunk];
	REQUIRE(trunkElem.children.size() == 1);
	CHECK(ir.elements[trunkElem.children[0]].type == PlantElementType::Hub);
}
