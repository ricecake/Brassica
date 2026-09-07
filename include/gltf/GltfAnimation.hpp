#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "gltf/GltfModel.hpp"

namespace brassica {

	class GltfAnimationEvaluator {
	public:
		static void EvaluateAnimation(
			const GltfAnimation& animation,
			float time,
			std::vector<GltfNode>& nodes
		);

		static void ComputeGlobalTransforms(std::vector<GltfNode>& nodes);

		static std::vector<glm::mat4> ComputeSkinningMatrices(
			const GltfSkin& skin,
			const std::vector<GltfNode>& nodes
		);
	};

} // namespace brassica
