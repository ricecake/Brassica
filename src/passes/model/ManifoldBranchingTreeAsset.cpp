#include "passes/model/ManifoldBranchingTreeAsset.hpp"

#include <algorithm>
#include <deque>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "passes/model/ManifoldSkeletalMeshBuilder.hpp"

namespace brassica {

	namespace {

		// Any reference axis not parallel to dir, for building a perpendicular basis.
		glm::vec3 PickPerpendicular(glm::vec3 dir) {
			glm::vec3 arbitrary = (std::abs(dir.y) < 0.99f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
			return glm::normalize(glm::cross(dir, arbitrary));
		}

		struct PendingNode {
			std::uint32_t jointIndex; // index into joints/specs, this node's own parent-to-be
			int           depth;
			glm::vec3     direction;
			float         length;
			float         radiusHigh; // this node's own end radius -- children continue from here
			float         rigidity;
		};

	} // namespace

	ModelAsset BuildProceduralBranchingTree(const TreeGenParams& params) {
		std::mt19937                          rng(params.seed);
		std::uniform_real_distribution<float> unit(0.0f, 1.0f);
		std::uniform_real_distribution<float> pitchDist(params.pitchDegreesMin, params.pitchDegreesMax);
		std::uniform_int_distribution<int>    branchCountDist(
			std::min(params.minBranchesPerNode, params.maxBranchesPerNode), std::max(params.minBranchesPerNode, params.maxBranchesPerNode)
		);

		std::vector<ManifoldJointSpec> joints;
		joints.reserve(params.maxJoints);

		glm::vec3 trunkDir(0.0f, 1.0f, 0.0f);
		joints.push_back(ManifoldJointSpec{
			glm::vec3(0.0f),
			trunkDir * params.trunkLength,
			params.trunkRadiusLow,
			params.trunkRadiusHigh,
			kModelNoSkeletonSentinel,
			params.baseRigidity,
		});

		std::deque<PendingNode> queue;
		queue.push_back(PendingNode{0u, 0, trunkDir, params.trunkLength, params.trunkRadiusHigh, params.baseRigidity});

		while (!queue.empty() && joints.size() < params.maxJoints) {
			PendingNode parent = queue.front();
			queue.pop_front();
			if (parent.depth >= params.maxDepth) {
				continue;
			}

			glm::vec3 perp = PickPerpendicular(parent.direction);
			int       childCount = branchCountDist(rng);
			for (int c = 0; c < childCount && joints.size() < params.maxJoints; ++c) {
				float     yaw = unit(rng) * glm::two_pi<float>();
				glm::quat yawRot = glm::angleAxis(yaw, parent.direction);
				glm::vec3 pitchAxis = yawRot * perp;
				float     pitch = glm::radians(pitchDist(rng));
				glm::quat pitchRot = glm::angleAxis(pitch, pitchAxis);
				glm::vec3 childDir = glm::normalize(pitchRot * parent.direction);

				float lengthJitter = 1.0f + params.lengthVariance * (unit(rng) * 2.0f - 1.0f);
				float childLength = std::max(0.05f, parent.length * params.lengthFalloff * lengthJitter);
				float childRadiusLow = parent.radiusHigh;
				float childRadiusHigh = std::max(0.01f, childRadiusLow * params.radiusFalloff);
				float childRigidity = std::max(0.05f, parent.rigidity - params.rigidityFalloffPerLevel);

				glm::vec3     childStart = joints[parent.jointIndex].end;
				glm::vec3     childEnd = childStart + childDir * childLength;
				std::uint32_t childIndex = static_cast<std::uint32_t>(joints.size());
				joints.push_back(ManifoldJointSpec{childStart, childEnd, childRadiusLow, childRadiusHigh, parent.jointIndex, childRigidity});

				queue.push_back(PendingNode{childIndex, parent.depth + 1, childDir, childLength, childRadiusHigh, childRigidity});
			}
		}

		return BuildSkinnedManifoldAsset(joints, params.circularSegments);
	}

} // namespace brassica
