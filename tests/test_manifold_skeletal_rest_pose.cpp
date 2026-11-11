#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vector>

#include "doctest/doctest.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"
#include "passes/model/ModelAsset.hpp"
#include "passes/model/ProceduralPlantGenerator.hpp"

using namespace brassica;

namespace {

	// Exactly shaders/model_instance.glsl's QuatTranslationScaleToMat4 -- the runtime pose walk
	// (model_live_eval.comp) reconstructs each joint's global transform with this formula from
	// ModelSkeletonJoint::restTranslation/restRotation/restScale, so a CPU-side test that wants
	// to prove those stored values are correct must reconstruct them the identical way, not
	// just trust BuildSkinnedManifoldAsset's own internal restGlobals array.
	glm::mat4 PoseFromRest(glm::vec3 t, glm::vec4 q, float s) {
		glm::quat quat(q.w, q.x, q.y, q.z); // glm::quat ctor order is (w,x,y,z)
		return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(quat) * glm::mat4(glm::mat3(s));
	}

} // namespace

// Regression test for a real, previously-shipped bug: ManifoldSkeletalMeshBuilder's skeleton
// bake once stored restTranslation/restRotation as *absolute* world quantities instead of
// parent-relative ones. That canceled out exactly at rest (restGlobals[j] * inverse(itself) is
// always identity, regardless of whether restGlobals[j] is geometrically correct) -- so static
// trees looked fine -- but broke completely the moment any wind-sway/live pose delta composed
// on top of it, since model_live_eval.comp's runtime walk assumes these are genuinely parent-
// relative. Confirmed via direct measurement before the fix: reconstructing each joint's global
// transform from the stored rest values and comparing against the real joint.start it should
// reproduce found mismatches up to several units on trees only ~2 units tall. This test
// reconstructs the hierarchy the exact same way the runtime shader does and checks it actually
// reproduces the real joint positions/orientations for a real branching tree, not a trivial
// straight one (the bug is invisible on anything with no rotated ancestors).
TEST_CASE("BuildSkinnedManifoldAsset's skeleton correctly reproduces real joint positions/orientations through a branching hierarchy") {
	PlantGenParams params;
	params.seed = 777;
	params.maxSegments = 24;
	params.segmentLength = 0.4f;
	params.directionJitterDegrees = 10.0f;
	params.branchChance = 1.0f; // force real branches every step -- the bug is invisible without them
	params.minBranchesPerNode = 2;
	params.maxBranchesPerNode = 3;
	params.branchPitchMinDegrees = 30.0f;
	params.branchPitchMaxDegrees = 70.0f;
	params.trunkRadiusLow = 0.2f;
	params.trunkRadiusHigh = 0.15f;
	params.maxJoints = 20;

	std::vector<ManifoldJointSpec> joints = BuildPlantJoints(params);
	REQUIRE(joints.size() > 5); // need real depth for the bug to show up at all

	ModelAsset asset = BuildSkinnedManifoldAsset(joints, params.circularSegments);
	REQUIRE(asset.skeleton.size() == joints.size());

	std::vector<glm::mat4> globalXform(joints.size());
	bool                   anyNonRootChecked = false;

	for (std::size_t j = 0; j < joints.size(); ++j) {
		const ModelSkeletonJoint& sj = asset.skeleton[j];
		glm::mat4                 parentGlobal =
			(sj.parentIndex == kModelNoSkeletonSentinel) ? glm::mat4(1.0f) : globalXform[sj.parentIndex];
		globalXform[j] = parentGlobal * PoseFromRest(sj.restTranslation, sj.restRotation, sj.restScale);

		glm::vec3 reconstructedOrigin = glm::vec3(globalXform[j][3]);
		CHECK(glm::distance(reconstructedOrigin, joints[j].start) < 1e-3f);

		// Hub/Flower/Leaf joints' "direction" is a near-zero placeholder offset (see
		// ConvertToJoints's own comment) -- only a real tube-shaped joint has a meaningful
		// growth direction to check the rotation column against.
		if (!joints[j].customShape && glm::length(joints[j].end - joints[j].start) > 1e-4f) {
			glm::vec3 expectedDir = glm::normalize(joints[j].end - joints[j].start);
			glm::vec3 reconstructedDir = glm::normalize(glm::vec3(globalXform[j] * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
			CHECK(glm::dot(expectedDir, reconstructedDir) > 0.999f);
		}

		if (sj.parentIndex != kModelNoSkeletonSentinel) {
			anyNonRootChecked = true;
		}
	}

	// If this seed/param combination never produced a joint with a real (non-root) parent, the
	// test above would pass trivially without ever exercising the bug -- make sure that's not
	// what happened.
	REQUIRE(anyNonRootChecked);
}
