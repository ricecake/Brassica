#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vector>

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
	// e1 absorbs e2's full span (endPosition extends to e2's own end) and the ControlPoint
	// records where the real bend was -- e2 itself is redundant once that happens (its own span
	// is already fully covered by e1's extended one, and the CP's children already point past
	// it to whatever e2's own children were), so it's removed, not kept alongside e1 as a
	// second, overlapping tube. One surviving tube, one ControlPoint marking the bend.
	REQUIRE(tubeCount == 1);
	CHECK(controlPointCount == 1);
	const PlantElement& merged = ir.elements[0];
	CHECK(merged.position == glm::vec3(0, 0, 0));
	CHECK(merged.endPosition == glm::vec3(1, 1, 0));
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

// Regression test for a real bug: ConsolidateTubes's "preserve as waypoint" branch (a real but
// not-perfectly-collinear bend) never marked the absorbed child as removed, unlike its sibling
// collinear-merge branch. The absorbed child survived, unreferenced by anyone going forward but
// still holding its own old `children` pointer -- and if *that* child's own index later came up
// again in the same pass's outer loop, it got independently reprocessed as its own "e1",
// producing a second, conflicting claim on its descendants. Across several bends in one long
// chain (exactly the shape a real procedural trunk has) this compounded into a genuine graph
// cycle: ConvertToJoints (ProceduralPlantGenerator.cpp) would loop through a handful of IR
// elements forever, only stopping once it hit maxJoints, producing far more ManifoldJointSpecs
// than the IR actually had elements, hiding the real terminal tube behind that cycle entirely
// (which is how a leaf/flower customShape attached at the true end could simply never be
// reached). Caught only because this test uses a 4-tube chain with two separate real bends --
// the existing 2-tube "preserves a real bend" test above has nothing for the surviving orphan to
// independently merge with, so it couldn't have caught this.
TEST_CASE("ConsolidateTubes on a long chain with multiple real bends produces no cycle and reaches the true end") {
	PlantIR ir;
	int     a = ir.AddTube({0, 0, 0}, {0, 1, 0}, 0.2f, 0.19f, 1.0f);
	int     b = ir.AddTube({0, 1, 0}, {0.6f, 1.6f, 0}, 0.19f, 0.17f, 1.0f, a);    // real bend vs a
	int     c = ir.AddTube({0.6f, 1.6f, 0}, {0.6f, 2.6f, 0}, 0.17f, 0.15f, 1.0f, b); // real bend vs b
	ir.AddTube({0.6f, 2.6f, 0}, {1.4f, 3.2f, 0}, 0.15f, 0.13f, 1.0f, c);             // real bend vs c, true end

	ProceduralPlantOptimizer::Optimize(ir);

	// Walk from the root via children, refusing to revisit an index -- if a cycle survived,
	// this loop would either revisit an index (caught explicitly) or never terminate (the
	// visited-set membership check below prevents that hang regardless).
	int                root = -1;
	for (std::size_t i = 0; i < ir.elements.size(); ++i) {
		if (ir.elements[i].parent == -1) {
			REQUIRE(root == -1); // exactly one root
			root = static_cast<int>(i);
		}
	}
	REQUIRE(root != -1);

	std::vector<bool> visited(ir.elements.size(), false);
	int                cur = root;
	int                steps = 0;
	glm::vec3          trueEnd{1.4f, 3.2f, 0.0f};
	bool               reachedTrueEnd = false;
	while (cur != -1) {
		REQUIRE_FALSE(visited[cur]); // a repeat visit means a cycle
		visited[cur] = true;
		++steps;
		REQUIRE(steps <= static_cast<int>(ir.elements.size())); // belt-and-suspenders vs a hang

		const PlantElement& e = ir.elements[cur];
		if (e.type == PlantElementType::Tube && e.endPosition == trueEnd) {
			reachedTrueEnd = true;
		}
		cur = e.children.empty() ? -1 : e.children[0];
	}
	CHECK(reachedTrueEnd);
}
