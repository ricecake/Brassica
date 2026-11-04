#include "model/ProceduralPlantGenerator.hpp"

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <manifold/manifold.h>

#include "model/ManifoldFlowerAsset.hpp"
#include "model/ManifoldSkeletalMeshBuilder.hpp"
#include "model/ProceduralPlantIR.hpp"
#include "model/ProceduralPlantOptimizer.hpp"

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
			queue.push_back(PendingSegment{trunkIdx, trunkDir, params.trunkRadiusHigh, params.baseRigidity});

			int segmentsGrown = 1;
			while (!queue.empty() && segmentsGrown < params.maxSegments) {
				PendingSegment parent = queue.front();
				queue.pop_front();
				glm::vec3 parentEnd = ir.elements[parent.irIndex].endPosition;
				glm::vec3 perp = PickPerpendicular(parent.direction);

				bool shouldBranch = unit(rng) < params.branchChance;
				int  childCount = shouldBranch ? branchCountDist(rng) : 1;

				for (int c = 0; c < childCount && segmentsGrown < params.maxSegments; ++c) {
					glm::vec3 childDir;
					if (shouldBranch) {
						float     yaw = unit(rng) * glm::two_pi<float>();
						glm::quat yawRot = glm::angleAxis(yaw, parent.direction);
						glm::vec3 pitchAxis = yawRot * perp;
						float     pitch = glm::radians(pitchDist(rng));
						glm::quat pitchRot = glm::angleAxis(pitch, pitchAxis);
						childDir = glm::normalize(pitchRot * parent.direction);
					} else {
						// Small continuation wander -- real branch points use the full pitch
						// range above; a straight-ish run only nudges slightly, which is what
						// gives the eventual consolidated joint a gentle curve instead of a hard
						// kink.
						float     jitterYaw = unit(rng) * glm::two_pi<float>();
						glm::quat yawRot = glm::angleAxis(jitterYaw, parent.direction);
						glm::vec3 jitterAxis = yawRot * perp;
						float     jitterAngle = glm::radians(params.directionJitterDegrees) * (unit(rng) * 2.0f - 1.0f);
						glm::quat jitterRot = glm::angleAxis(jitterAngle, jitterAxis);
						childDir = glm::normalize(jitterRot * parent.direction);
					}

					float lengthJitter = 1.0f + params.segmentLengthVariance * (unit(rng) * 2.0f - 1.0f);
					float childLength = std::max(0.05f, params.segmentLength * lengthJitter);
					float childRadiusHigh = std::max(0.01f, parent.radiusHigh * params.radiusFalloffPerSegment);
					float childRigidity = std::max(0.05f, parent.rigidity - params.rigidityFalloffPerSegment);

					glm::vec3 childStart = parentEnd;
					glm::vec3 childEnd = childStart + childDir * childLength;
					int       childIdx =
						ir.AddTube(childStart, childEnd, parent.radiusHigh, childRadiusHigh, childRigidity, parent.irIndex);

					queue.push_back(PendingSegment{childIdx, childDir, childRadiusHigh, childRigidity});
					++segmentsGrown;
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
				// A flower is always a leaf -- no recursion.
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
		ProceduralPlantOptimizer::Optimize(ir);

		std::vector<ManifoldJointSpec> joints;
		for (int i = 0; i < static_cast<int>(ir.elements.size()); ++i) {
			if (ir.elements[i].parent == -1) {
				ConvertToJoints(ir, i, kModelNoSkeletonSentinel, glm::vec3(0.0f, 1.0f, 0.0f), params, joints);
			}
		}
		return joints;
	}

	ModelAsset BuildProceduralPlant(const PlantGenParams& params) {
		return BuildSkinnedManifoldAsset(BuildPlantJoints(params), params.circularSegments);
	}

} // namespace brassica
