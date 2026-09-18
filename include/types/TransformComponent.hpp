#pragma once

#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	struct TransformComponent {
		glm::vec3 position{0.0f};
		glm::vec3 rotation{0.0f}; // Euler angles in radians
		glm::vec3 scale{1.0f};

		[[nodiscard]] glm::mat4 GetTransformMatrix() const {
			glm::mat4 transform = glm::translate(glm::mat4(1.0f), position);
			transform = glm::rotate(transform, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
			transform = glm::rotate(transform, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
			transform = glm::rotate(transform, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
			transform = glm::scale(transform, scale);
			return transform;
		}
	};

} // namespace brassica
