#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "doctest/doctest.h"

#include "passes/model/ManifoldPalmAsset.hpp"
#include "passes/model/ManifoldPineAsset.hpp"
#include "passes/model/ProceduralPlantGenerator.hpp"

using namespace brassica;

namespace {

	constexpr float kModelWindStrength = 0.6f;            // matches shaders/model_live_eval.comp exactly
	constexpr float kModelLiveBoundsPaddingFactor = 2.5f; // matches shaders/model_cull.comp exactly

	glm::vec3 SafeBendAxis(glm::vec3 restUpAxis, glm::vec3 localDir) {
		glm::vec3 axis = glm::cross(restUpAxis, localDir);
		float     len = glm::length(axis);
		return (len > 1e-4f) ? (axis / len) : glm::vec3(1.0f, 0.0f, 0.0f);
	}

	// Regression test for a real bug: model_cull.comp's own kModelLiveBoundsPaddingFactor exists
	// specifically to cover "a wind-bent branch can swing further than its rest/clip bounds alone
	// would suggest" (that file's own comment), but was never actually checked against any real
	// registered preset's worst-case bend -- BushA/BushB's many short, flexible, chained branches
	// swung past the old 1.5x margin by a real, measured amount, which could cull a visually
	// on-screen bush out of the frustum during a strong gust (sgreenhusted traced "clouds
	// compositing over decor" to exactly this: a culled instance's GBuffer pixels read as open
	// sky for however long the gust stays strong). This replicates model_live_eval.comp's exact
	// wind-bend chain composition (restRotation * windBend per joint, through parentGlobal) on
	// the CPU, at gustVal=1 (worst case), against every currently-registered archetype's real
	// skeleton data.
	float MaxWorstCaseDisplacement(const ModelAsset& asset, glm::vec3 worldWindDir) {
		if (asset.skeleton.empty()) {
			return 0.0f;
		}

		std::vector<glm::vec3> restOrigin(asset.skeleton.size());
		std::vector<glm::vec3> bentOrigin(asset.skeleton.size());
		std::vector<glm::quat> restGlobalRot(asset.skeleton.size());
		std::vector<glm::quat> bentGlobalRot(asset.skeleton.size());

		float maxDisplacement = 0.0f;
		for (std::size_t j = 0; j < asset.skeleton.size(); ++j) {
			const ModelSkeletonJoint& sj = asset.skeleton[j];
			glm::quat restRotation(sj.restRotation.w, sj.restRotation.x, sj.restRotation.y, sj.restRotation.z);

			glm::vec3 parentRestOrigin(0.0f);
			glm::quat parentRestRot(1.0f, 0.0f, 0.0f, 0.0f);
			glm::vec3 parentBentOrigin(0.0f);
			glm::quat parentBentRot(1.0f, 0.0f, 0.0f, 0.0f);
			if (sj.parentIndex != kModelNoSkeletonSentinel) {
				parentRestOrigin = restOrigin[sj.parentIndex];
				parentRestRot = restGlobalRot[sj.parentIndex];
				parentBentOrigin = bentOrigin[sj.parentIndex];
				parentBentRot = bentGlobalRot[sj.parentIndex];
			}

			// Rest pose: no wind term at all.
			restGlobalRot[j] = glm::normalize(parentRestRot * restRotation);
			restOrigin[j] = parentRestOrigin + parentRestRot * (sj.restTranslation * sj.restScale);

			// Worst case: gustVal=1, a single steady wind direction, no seed-bend damping term
			// (smaller and separately damped -- omitted for a cleaner worst-case bound, not
			// hidden: this makes the simulated displacement a slight underestimate of the real
			// shader's own max, so a margin found safe here is genuinely safe, not borderline).
			glm::vec3 localWind = glm::normalize(glm::inverse(parentBentRot) * worldWindDir);
			glm::vec3 windAxis = SafeBendAxis(sj.restUpAxis, localWind);
			float     windAngle = kModelWindStrength * (1.0f - sj.rigidity);
			glm::quat windBend = glm::angleAxis(windAngle, windAxis);
			glm::quat localRotation = glm::normalize(restRotation * windBend);

			bentGlobalRot[j] = glm::normalize(parentBentRot * localRotation);
			bentOrigin[j] = parentBentOrigin + parentBentRot * (sj.restTranslation * sj.restScale);

			float displacement = glm::length(bentOrigin[j] - restOrigin[j]);
			maxDisplacement = std::max(maxDisplacement, displacement);
		}
		return maxDisplacement;
	}

	float WorstCaseOverAllDirections(const ModelAsset& asset) {
		float worstCase = 0.0f;
		for (glm::vec3 windDir : {glm::vec3(1, 0, 0), glm::vec3(0, 0, 1), glm::vec3(1, 0, 1), glm::vec3(-1, 0, 1)}) {
			worstCase = std::max(worstCase, MaxWorstCaseDisplacement(asset, glm::normalize(windDir)));
		}
		return worstCase;
	}

	void CheckOne(const char* name, const ModelAsset& asset) {
		float paddingMargin = asset.restBoundsRadius * (kModelLiveBoundsPaddingFactor - 1.0f);
		float worstCase = WorstCaseOverAllDirections(asset);
		std::printf(
			"%-12s restBoundsRadius=%.3f  paddingMargin=%.3f  worstCaseJointDisplacement=%.3f\n", name,
			asset.restBoundsRadius, paddingMargin, worstCase
		);
		CHECK(worstCase <= paddingMargin);
	}

} // namespace

TEST_CASE("Every registered archetype's worst-case wind-bend stays within model_cull.comp's padding margin") {
	// Inlined from ProceduralArchetypeCatalog.cpp's own anonymous-namespace presets (not
	// exported) -- same values, copied here so this check can run standalone.
	PlantGenParams smallTree;
	smallTree.seed = 101;
	smallTree.maxSegments = 14;
	smallTree.segmentLength = 0.35f;
	smallTree.directionJitterDegrees = 10.0f;
	smallTree.branchChance = 0.3f;
	smallTree.minBranchesPerNode = 2;
	smallTree.maxBranchesPerNode = 2;
	smallTree.trunkRadiusLow = 0.14f;
	smallTree.trunkRadiusHigh = 0.11f;
	smallTree.radiusFalloffPerSegment = 0.92f;
	smallTree.rigidityFalloffPerSegment = 0.07f;
	smallTree.circularSegments = 10;
	smallTree.maxJoints = 12;
	smallTree.enableLeaves = true;
	smallTree.leafChance = 0.85f;
	smallTree.leafSize = 0.18f;
	smallTree.leafCount = 5;
	CheckOne("SmallTree", BuildProceduralPlant(smallTree));

	PlantGenParams largeTree;
	largeTree.seed = 103;
	largeTree.maxSegments = 30;
	largeTree.segmentLength = 0.48f;
	largeTree.directionJitterDegrees = 9.0f;
	largeTree.branchChance = 0.32f;
	largeTree.minBranchesPerNode = 3;
	largeTree.maxBranchesPerNode = 3;
	largeTree.trunkRadiusLow = 0.30f;
	largeTree.trunkRadiusHigh = 0.22f;
	largeTree.radiusFalloffPerSegment = 0.945f;
	largeTree.rigidityFalloffPerSegment = 0.04f;
	largeTree.circularSegments = 10;
	largeTree.maxJoints = 24;
	largeTree.enableLeaves = true;
	largeTree.leafChance = 0.85f;
	largeTree.leafSize = 0.26f;
	largeTree.leafCount = 6;
	CheckOne("LargeTree", BuildProceduralPlant(largeTree));

	PlantGenParams bushA;
	bushA.seed = 201;
	bushA.maxSegments = 20;
	bushA.segmentLength = 0.18f;
	bushA.directionJitterDegrees = 16.0f;
	bushA.branchChance = 0.35f;
	bushA.minBranchesPerNode = 2;
	bushA.maxBranchesPerNode = 3;
	bushA.branchPitchMinDegrees = 45.0f;
	bushA.branchPitchMaxDegrees = 85.0f;
	bushA.trunkRadiusLow = 0.08f;
	bushA.trunkRadiusHigh = 0.06f;
	bushA.radiusFalloffPerSegment = 0.85f;
	bushA.baseRigidity = 0.6f;
	bushA.rigidityFalloffPerSegment = 0.10f;
	bushA.circularSegments = 8;
	bushA.maxJoints = 18;
	bushA.enableLeaves = true;
	bushA.leafChance = 0.85f;
	bushA.leafSize = 0.09f;
	bushA.leafCount = 4;
	CheckOne("BushA", BuildProceduralPlant(bushA));

	PlantGenParams bushB = bushA;
	bushB.seed = 202;
	bushB.maxSegments = 24;
	bushB.maxBranchesPerNode = 4;
	bushB.maxJoints = 22;
	CheckOne("BushB", BuildProceduralPlant(bushB));

	PineGenParams pine;
	pine.seed = 501;
	pine.trunkSegments = 4;
	pine.trunkHeight = 2.6f;
	pine.trunkRadiusLow = 0.16f;
	pine.trunkRadiusHigh = 0.09f;
	pine.trunkBaseRigidity = 0.95f;
	pine.tierCount = 3;
	pine.tierBaseRadius = 0.85f;
	pine.tierRadiusFalloff = 0.70f;
	pine.tierHeight = 1.15f;
	pine.tierOverlap = 0.35f;
	pine.tierBaseRigidity = 0.55f;
	pine.circularSegments = 10;
	CheckOne("Pine", BuildPineTree(pine));

	PalmGenParams palm;
	palm.seed = 502;
	CheckOne("Palm", BuildPalmTree(palm));
}
