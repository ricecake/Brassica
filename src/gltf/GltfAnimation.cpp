#include "gltf/GltfAnimation.hpp"
#include <cmath>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	void GltfAnimationEvaluator::EvaluateAnimation(
		const GltfAnimation& animation,
		float time,
		std::vector<GltfNode>& nodes
	) {
		if (animation.channels.empty() || animation.duration <= 0.0f) return;

		// Loop time within duration
		float animTime = std::fmod(time, animation.duration);
		if (animTime < 0.0f) animTime += animation.duration;

		for (const auto& channel : animation.channels) {
			if (channel.targetNode < 0 || channel.targetNode >= static_cast<int32_t>(nodes.size())) continue;
			if (channel.samplersInput.size() < 2) continue;

			auto& node = nodes[channel.targetNode];

			// Find keyframe interval
			size_t k = 0;
			while (k < channel.samplersInput.size() - 1 && channel.samplersInput[k + 1] <= animTime) {
				k++;
			}

			size_t kNext = std::min(k + 1, channel.samplersInput.size() - 1);
			float t0 = channel.samplersInput[k];
			float t1 = channel.samplersInput[kNext];
			float dt = t1 - t0;
			float factor = (dt > 0.00001f) ? (animTime - t0) / dt : 0.0f;
			factor = std::clamp(factor, 0.0f, 1.0f);

			if (channel.path == GltfTargetPath::Translation) {
				glm::vec3 val0 = glm::vec3(channel.samplersOutput[k]);
				glm::vec3 val1 = glm::vec3(channel.samplersOutput[kNext]);
				node.translation = (channel.interpolation == GltfInterpolationType::Step) ? val0 : glm::mix(val0, val1, factor);
			} else if (channel.path == GltfTargetPath::Rotation) {
				glm::quat q0 = glm::quat(channel.samplersOutput[k].w, channel.samplersOutput[k].x, channel.samplersOutput[k].y, channel.samplersOutput[k].z);
				glm::quat q1 = glm::quat(channel.samplersOutput[kNext].w, channel.samplersOutput[kNext].x, channel.samplersOutput[kNext].y, channel.samplersOutput[kNext].z);
				node.rotation = (channel.interpolation == GltfInterpolationType::Step) ? q0 : glm::slerp(q0, q1, factor);
			} else if (channel.path == GltfTargetPath::Scale) {
				glm::vec3 val0 = glm::vec3(channel.samplersOutput[k]);
				glm::vec3 val1 = glm::vec3(channel.samplersOutput[kNext]);
				node.scale = (channel.interpolation == GltfInterpolationType::Step) ? val0 : glm::mix(val0, val1, factor);
			}

			glm::mat4 T = glm::translate(glm::mat4(1.0f), node.translation);
			glm::mat4 R = glm::mat4_cast(node.rotation);
			glm::mat4 S = glm::scale(glm::mat4(1.0f), node.scale);
			node.localMatrix = T * R * S;
		}
	}

	void GltfAnimationEvaluator::ComputeGlobalTransforms(std::vector<GltfNode>& nodes) {
		auto computeNodeGlobal = [&](auto& self, int32_t nodeIdx, const glm::mat4& parentGlobal) -> void {
			if (nodeIdx < 0 || nodeIdx >= static_cast<int32_t>(nodes.size())) return;
			auto& node = nodes[nodeIdx];
			node.globalMatrix = parentGlobal * node.localMatrix;

			for (int32_t child : node.children) {
				self(self, child, node.globalMatrix);
			}
		};

		for (size_t i = 0; i < nodes.size(); ++i) {
			if (nodes[i].parent == -1) {
				computeNodeGlobal(computeNodeGlobal, static_cast<int32_t>(i), glm::mat4(1.0f));
			}
		}
	}

	std::vector<glm::mat4> GltfAnimationEvaluator::ComputeSkinningMatrices(
		const GltfSkin& skin,
		const std::vector<GltfNode>& nodes
	) {
		std::vector<glm::mat4> jointMatrices(skin.joints.size(), glm::mat4(1.0f));

		for (size_t i = 0; i < skin.joints.size(); ++i) {
			int32_t jointNodeIdx = skin.joints[i];
			if (jointNodeIdx >= 0 && jointNodeIdx < static_cast<int32_t>(nodes.size())) {
				const auto& node = nodes[jointNodeIdx];
				glm::mat4 ibm = (i < skin.inverseBindMatrices.size()) ? skin.inverseBindMatrices[i] : glm::mat4(1.0f);
				jointMatrices[i] = node.globalMatrix * ibm;
			}
		}

		return jointMatrices;
	}

} // namespace brassica
