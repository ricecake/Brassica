#include "model/ManifoldTreeAsset.hpp"

#include <array>

#include "model/ManifoldSkeletalMeshBuilder.hpp"

namespace brassica {

	ModelAsset BuildManifoldTestTree() {
		// Joint 0: trunk, straight up from the origin. Joint 1: a branch angled off
		// partway up the trunk. Trunk stays rigid; the branch is the one Stage 4's
		// ModelLiveEvalNode should visibly bend under wind.
		const std::array<ManifoldJointSpec, 2> joints{
			ManifoldJointSpec{glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 2.0f, 0.0f), 0.25f, 0.18f, kModelNoSkeletonSentinel, 0.9f},
			ManifoldJointSpec{glm::vec3(0.0f, 1.3f, 0.0f), glm::vec3(0.9f, 2.3f, 0.0f), 0.14f, 0.05f, 0u, 0.15f},
		};
		return BuildSkinnedManifoldAsset(joints);
	}

} // namespace brassica
