#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "gltf/GltfModel.hpp"

namespace brassica {

	struct TransformComponent {
		glm::vec3 position{0.0f, 0.0f, 0.0f};
		glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
		glm::vec3 scale{1.0f, 1.0f, 1.0f};

		glm::mat4 GetWorldMatrix() const {
			glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
			glm::mat4 R = glm::mat4_cast(rotation);
			glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
			return T * R * S;
		}

		void SetEulerAngles(const glm::vec3& eulerAnglesDegrees) {
			glm::vec3 radians = glm::radians(eulerAnglesDegrees);
			rotation = glm::quat(radians);
		}

		glm::vec3 GetEulerAngles() const {
			return glm::degrees(glm::eulerAngles(rotation));
		}
	};

	struct GltfModelComponent {
		std::shared_ptr<GltfModel> model{nullptr};
		bool visible{true};
	};

	struct GltfAnimationComponent {
		int32_t animationIndex{0};
		float time{0.0f};
		float speed{1.0f};
		bool playing{true};
		bool loop{true};
	};

	struct AvatarComponent {
		glm::vec3 offset{0.0f, -0.5f, 2.0f};
		glm::vec3 rotationOffset{0.0f, 180.0f, 0.0f};
	};

} // namespace brassica
