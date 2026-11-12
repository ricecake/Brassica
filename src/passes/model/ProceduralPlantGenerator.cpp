#include "passes/model/ProceduralPlantGenerator.hpp"

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <manifold/manifold.h>

#include "passes/model/ManifoldFlowerAsset.hpp"
#include "passes/model/ManifoldLeafAsset.hpp"
#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"
#include "passes/model/ProceduralPlantIR.hpp"
#include "passes/model/ProceduralPlantOptimizer.hpp"

namespace brassica {

	namespace {

		glm::vec3 PickPerpendicular(glm::vec3 dir) {
			glm::vec3 arbitrary = (std::abs(dir.y) < 0.99f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
			return glm::normalize(glm::cross(dir, arbitrary));
		}

		struct PendingSegment {
			int       irIndex; // this segment's own Tube index in ir.elements
			glm::vec3 direction;
			float     radiusHigh;
			float     rigidity;
			int       segmentsSinceBranch; // see PlantGenParams::minSegmentsBetweenBranches
			int       chainDepth;
		};

		// Grows the raw IR: each queue entry is one short segment, branching (2-3 children) with
		// probability branchChance or continuing (1 jittered child) otherwise. A run of
		// continuations is what ProceduralPlantOptimizer later consolidates into one curved
		// joint -- the jitter is what gives that consolidated joint an actual curve to follow.
		void GrowPlantIR(PlantIR& ir, const PlantGenParams& params, std::mt19937& rng) {
			std::uniform_real_distribution<float> unit(0.0f, 1.0f);
			std::uniform_real_distribution<float> pitchDist(params.branchPitchMinDegrees, params.branchPitchMaxDegrees);
			std::uniform_int_distribution<int>    branchCountDist(
				std::min(params.minBranchesPerNode, params.maxBranchesPerNode),
				std::max(params.minBranchesPerNode, params.maxBranchesPerNode)
			);

			glm::vec3 trunkDir(0.0f, 1.0f, 0.0f);
			int       trunkIdx = ir.AddTube(
				glm::vec3(0.0f), trunkDir * params.segmentLength, params.trunkRadiusLow, params.trunkRadiusHigh, params.baseRigidity
			);

			std::deque<PendingSegment> queue;
			// The root starts already "cooled down" (eligible to branch immediately), matching
			// the old no-cooldown behavior at the trunk base.
			queue.push_back(PendingSegment{
				// trunkIdx, trunkDir, params.trunkRadiusHigh, params.baseRigidity, params.minSegmentsBetweenBranches
				trunkIdx, trunkDir, params.trunkRadiusHigh, params.baseRigidity, params.minSegmentsBetweenBranches, 1
			});

			while (!queue.empty()) {
				PendingSegment parent = queue.front();
				queue.pop_front();

				if (parent.chainDepth >= params.maxSegments) {
					continue;
				}

				glm::vec3 parentEnd = ir.elements[parent.irIndex].endPosition;
				glm::vec3 perp = PickPerpendicular(parent.direction);

				// A branch child can't itself branch again until it's grown at least
				// minSegmentsBetweenBranches plain continuations -- without this, two branch
				// points can stack directly against each other with no straight tube run between
				// them, which is what was producing crowded/colliding branch clusters at high
				// branchChance (BushVariantA/B's 0.5).
				bool eligibleToBranch = parent.segmentsSinceBranch >= params.minSegmentsBetweenBranches;
				bool shouldBranch = eligibleToBranch && (unit(rng) < params.branchChance);
				int  childCount = shouldBranch ? branchCountDist(rng) : 1;

				float depthFactor = std::min(1.0f, static_cast<float>(parent.chainDepth) / (params.maxSegments * 0.4f));
				float baseYaw = unit(rng) * glm::two_pi<float>();
				for (int c = 0; c < childCount; ++c) {
					glm::vec3 childDir;
					if (shouldBranch) {
						// Distribute branches evenly (e.g. 180 deg apart for 2, 120 deg apart for 3)
						float yaw = baseYaw + (glm::two_pi<float>() * static_cast<float>(c) / static_cast<float>(childCount));
						yaw += (unit(rng) - 0.5f) * 0.2f; // Slight natural jitter so it isn't perfectly mathematical

						glm::quat yawRot = glm::angleAxis(yaw, parent.direction);
						glm::vec3 pitchAxis = yawRot * perp;

						// Suppress pitch low on the trunk so early branches grow mostly upward
						float pitch = glm::radians(pitchDist(rng)) * depthFactor;
						glm::quat pitchRot = glm::angleAxis(pitch, pitchAxis);
						glm::vec3 rawDir = pitchRot * parent.direction;

						// Keep the upward pull to shape the canopy
						float upwardPull = 0.5f * (static_cast<float>(parent.radiusHigh) / params.segmentLength);
						childDir = glm::normalize(rawDir + glm::vec3(0.0f, 1.0f, 0.0f) * upwardPull);
					} else {
						float jitterYaw = unit(rng) * glm::two_pi<float>();
						glm::quat yawRot = glm::angleAxis(jitterYaw, parent.direction);
						glm::vec3 jitterAxis = yawRot * perp;

						// Multiply the jitter angle by depthFactor to prevent the trunk from kinking early on
						float jitterAngle = glm::radians(params.directionJitterDegrees) * (unit(rng) * 2.0f - 1.0f) * depthFactor;
						glm::quat jitterRot = glm::angleAxis(jitterAngle, jitterAxis);
						childDir = glm::normalize(jitterRot * parent.direction);
					}

					float lengthJitter = 1.0f + params.segmentLengthVariance * (unit(rng) * 2.0f - 1.0f);
					float childLength = std::max(0.05f, params.segmentLength * lengthJitter);
					float childRadiusHigh = std::max(0.01f, parent.radiusHigh * params.radiusFalloffPerSegment);

					float depthRatio = static_cast<float>(parent.chainDepth) / static_cast<float>(params.maxSegments);
					float childRigidity = glm::mix(params.baseRigidity, 0.05f, std::min(1.0f, depthRatio));

					glm::vec3 childStart = parentEnd;
					glm::vec3 childEnd = childStart + childDir * childLength;
					int       childIdx =
						ir.AddTube(childStart, childEnd, parent.radiusHigh, childRadiusHigh, childRigidity, parent.irIndex);

					int childSegmentsSinceBranch = shouldBranch ? 0 : parent.segmentsSinceBranch + 1;
					queue.push_back(PendingSegment{
						childIdx, childDir, childRadiusHigh, childRigidity, childSegmentsSinceBranch, parent.chainDepth + 1
					});
				}
			}
		}

		// A terminal tube (grew no children at all, whether from a real dead end or just running
		// out of segment budget) gets a chance at a flower, added before optimization so the
		// usual consolidation/junction rules apply uniformly to it too.
		void AddFlowers(PlantIR& ir, const PlantGenParams& params, std::mt19937& rng) {
			std::uniform_real_distribution<float> unit(0.0f, 1.0f);
			int                                   originalCount = static_cast<int>(ir.elements.size());
			for (int i = 0; i < originalCount; ++i) {
				if (ir.elements[i].type != PlantElementType::Tube || !ir.elements[i].children.empty()) {
					continue;
				}
				if (unit(rng) >= params.flowerChance) {
					continue;
				}
				int variant = static_cast<int>(unit(rng) * 4.0f) % 4;
				ir.AddFlower(ir.elements[i].endPosition, params.flowerSize, variant, 1.0f, i);
			}
		}

		// A terminal tube that didn't already grow a flower (AddFlowers, if enabled, runs first
		// and leaves flowered tips with a real child -- the same "no children" check below
		// naturally skips them) gets a chance at a leaf tuft instead.
void AddLeaves(PlantIR& ir, const PlantGenParams& params, std::mt19937& rng) {
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);
	int                                   originalCount = static_cast<int>(ir.elements.size());

	// Determine dynamic leaf size based on the highest point in the tree
	float maxTreeHeight = 0.1f;
	for (int i = 0; i < originalCount; ++i) {
		maxTreeHeight = std::max(maxTreeHeight, ir.elements[i].endPosition.y);
	}
	// Scale cluster size to 15% of the tree's total height, using params.leafSize as a baseline modifier
	float dynamicLeafSize = maxTreeHeight * 0.15f * params.leafSize;

	for (int i = 0; i < originalCount; ++i) {
		if (ir.elements[i].type != PlantElementType::Tube || !ir.elements[i].children.empty()) {
			continue;
		}
		if (unit(rng) >= params.leafChance) {
			continue;
		}
		ir.AddLeaf(
			ir.elements[i].endPosition, dynamicLeafSize, params.leafCount, ir.elements[i].endRadius, 0.3f, i
		);
	}
}

		// Post-optimization invariant (ProceduralPlantOptimizer.cpp's own HandleJunctions):
		// every Tube/ControlPoint has 0 or 1 children -- multi-child branch points only ever
		// exist on Hub elements now. Walks forward through a chain of ControlPoints starting at
		// idx's own child, collecting each one's position as a curve waypoint, and returns the
		// first non-ControlPoint descendant (a Tube/Hub/Flower), or -1 if the chain simply ends.
		int SkipControlPoints(const PlantIR& ir, int idx, std::vector<glm::vec3>& waypoints) {
			const PlantElement& e = ir.elements[idx];
			if (e.children.empty()) {
				return -1;
			}
			int childIdx = e.children[0];
			if (ir.elements[childIdx].type == PlantElementType::ControlPoint) {
				waypoints.push_back(ir.elements[childIdx].position);
				return SkipControlPoints(ir, childIdx, waypoints);
			}
			return childIdx;
		}

		// Converts the optimized IR into ManifoldJointSpecs, enforcing maxJoints as a hard cap
		// (truncates the tree rather than exceeding it, same "harmless, bounded" convention
		// TreeGenParams::maxJoints already uses).
		void ConvertToJoints(
			const PlantIR& ir, int idx, std::uint32_t parentJoint, glm::vec3 inheritedDir, const PlantGenParams& params,
			std::vector<ManifoldJointSpec>& joints
		) {
			if (joints.size() >= params.maxJoints) {
				return;
			}
			const PlantElement& e = ir.elements[idx];

			if (e.type == PlantElementType::Tube) {
				std::vector<glm::vec3> waypoints;
				int                    nextIdx = SkipControlPoints(ir, idx, waypoints);

				ManifoldJointSpec spec{};
				spec.start = e.position;
				spec.end = e.endPosition;
				spec.radiusLow = e.radius;
				spec.radiusHigh = e.endRadius;
				spec.rigidity = e.rigidity;
				spec.parentIndex = parentJoint;
				spec.curveWaypoints = waypoints;

				auto myJoint = static_cast<std::uint32_t>(joints.size());
				joints.push_back(spec);

				glm::vec3 dir = glm::normalize(e.endPosition - e.position);
				if (nextIdx != -1) {
					ConvertToJoints(ir, nextIdx, myJoint, dir, params, joints);
				}
			} else if (e.type == PlantElementType::Hub) {
				ManifoldJointSpec spec{};
				spec.start = e.position;
				spec.end = e.position + inheritedDir * 0.01f; // tiny offset, just for orientation -- a sphere looks the same regardless
				spec.rigidity = 1.0f;
				spec.parentIndex = parentJoint;
				float radius = e.radius;
				spec.customShape = [radius]() { return manifold::Manifold::Sphere(static_cast<double>(radius), 12); };

				auto myJoint = static_cast<std::uint32_t>(joints.size());
				joints.push_back(spec);

				for (int child : e.children) {
					ConvertToJoints(ir, child, myJoint, inheritedDir, params, joints);
				}
			} else if (e.type == PlantElementType::Flower) {
				ManifoldJointSpec spec{};
				spec.start = e.position;
				spec.end = e.position + inheritedDir * 0.01f;
				spec.rigidity = e.rigidity;
				spec.parentIndex = parentJoint;
				float         size = e.radius;
				FlowerProfile profile = static_cast<FlowerProfile>(e.variant);
				spec.customShape = [size, profile]() { return BuildFlowerHead(size, profile); };
				joints.push_back(spec);
				// A flower is always terminal -- no recursion.
			} else if (e.type == PlantElementType::Leaf) {
				ManifoldJointSpec spec{};
				spec.start = e.position;
				spec.end = e.position + inheritedDir * 0.01f;
				spec.rigidity = e.rigidity;
				spec.parentIndex = parentJoint;
				float         size = e.radius;
				int           leafCount = e.variant;
				float         attachRadius = e.endRadius;
				// Deterministic per joint (not per plant), so the same seed always regrows the
				// same tuft shape -- idx is this element's own stable IR index.
				std::uint32_t leafSeed = params.seed * 2654435761u + static_cast<std::uint32_t>(idx);
				float         widthRatio = params.leafWidthRatio;
				bool          ballCluster = params.leafBallCluster;
				spec.customShape = [size, leafCount, leafSeed, widthRatio, attachRadius, ballCluster]() {
					return ballCluster ? BuildLeafBallCluster(size, leafCount, leafSeed, widthRatio, attachRadius)
										: BuildLeafCluster(size, leafCount, leafSeed, widthRatio, attachRadius);
				};
				if (params.leafMaterialIndex.has_value()) {
					spec.materialIndex = params.leafMaterialIndex;
				}
				joints.push_back(spec);
				// A leaf tuft is always terminal -- no recursion.
			}
		}

	} // namespace

std::vector<ManifoldJointSpec> BuildPlantJoints(const PlantGenParams& params) {
	std::mt19937 rng(params.seed);
	PlantIR ir;
	GrowPlantIR(ir, params, rng);
	if (params.enableFlowers) {
		AddFlowers(ir, params, rng);
	}

	// REMOVED: AddLeaves(ir, params, rng);

	ProceduralPlantOptimizer::Optimize(ir);

	std::vector<ManifoldJointSpec> joints;
	for (int i = 0; i < static_cast<int>(ir.elements.size()); ++i) {
		if (ir.elements[i].parent == -1) {
			ConvertToJoints(ir, i, kModelNoSkeletonSentinel, glm::vec3(0.0f, 1.0f, 0.0f), params, joints);
		}
	}

	if (params.enableLeaves) {
		std::vector<glm::vec3> terminalPoints;
		std::vector<std::uint32_t> terminalJoints;
		float maxTreeHeight = 0.1f;

		// 1. Find terminal joints and calculate the tree's maximum height
		for (std::uint32_t i = 0; i < joints.size(); ++i) {
			maxTreeHeight = std::max(maxTreeHeight, joints[i].end.y);

			bool isParent = false;
			for (const auto& j : joints) {
				if (j.parentIndex == i) {
					isParent = true;
					break;
				}
			}
			if (!isParent) {
				terminalJoints.push_back(i);
				terminalPoints.push_back(joints[i].end);
			}
		}

		if (!terminalPoints.empty()) {
			ManifoldJointSpec canopySpec{};
			canopySpec.parentIndex = 0;
			canopySpec.start = glm::vec3(0.0f, 0.0f, 0.0f);
			canopySpec.end   = glm::vec3(0.0f, 0.0f, 1.0f);
			canopySpec.rigidity = params.baseRigidity;
			canopySpec.blendJoints = terminalJoints; // Enables soft-skinning

			if (params.leafMaterialIndex.has_value()) {
				canopySpec.materialIndex = params.leafMaterialIndex;
			}

			// 2. Dynamically scale leaf size based on generated tree height
			float dynamicLeafSize = maxTreeHeight * 0.15f * params.leafSize;
			std::uint32_t seed = params.seed;

			canopySpec.customShape = [terminalPoints, dynamicLeafSize, seed]() {
				return BuildGlobalCanopyHull(terminalPoints, dynamicLeafSize, seed);
			};

			joints.push_back(canopySpec);
		}
	}

	return joints;
}

	ModelAsset BuildProceduralPlant(const PlantGenParams& params) {
		return BuildSkinnedManifoldAsset(BuildPlantJoints(params), params.circularSegments, params.materialIndex);
	}

} // namespace brassica
