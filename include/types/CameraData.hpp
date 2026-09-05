#pragma once

#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace brassica {

	struct CameraData {
		// Position and Orientation
		glm::vec3 position{0.0f, 15.0f, 30.0f};
		glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f}; // Unit quaternion

		// Euler angles in radians
		float pitch{-0.4636476f}; // ~ -26.57 deg (looking down towards origin from (0, 15, 30))
		float yaw{-1.5707963f};   // ~ -90 deg (looking along -Z)
		float roll{0.0f};         // radians

		// FOV data and frustum settings
		float fov{1.2f};           // vertical FOV in radians (~68.75 deg)
		float nearPlane{0.1f};
		float farPlane{500.0f};
		float aspectRatio{16.0f / 9.0f};

		// Speed and speed limits
		float speed{10.0f};
		float defaultSpeed{10.0f};
		float minSpeed{1.0f};
		float maxSpeed{100.0f};
		float speedStep{5.0f};

		// Control state
		bool isCaptured{false};

		// Computed Matrices and Frustum Planes
		glm::mat4 viewMatrix{1.0f};
		glm::mat4 projMatrix{1.0f};
		glm::mat4 viewProjMatrix{1.0f};
		std::array<glm::vec4, 6> frustumPlanes{}; // Left, Right, Bottom, Top, Near, Far

		// Direction vectors
		glm::vec3 GetForward() const {
			return orientation * glm::vec3(0.0f, 0.0f, -1.0f);
		}

		glm::vec3 GetUp() const {
			return orientation * glm::vec3(0.0f, 1.0f, 0.0f);
		}

		glm::vec3 GetRight() const {
			return orientation * glm::vec3(1.0f, 0.0f, 0.0f);
		}

		void UpdateOrientation() {
			glm::quat qYaw = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::quat qPitch = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
			glm::quat qRoll = glm::angleAxis(roll, glm::vec3(0.0f, 0.0f, 1.0f));
			orientation = qYaw * qPitch * qRoll;
		}

		void UpdateMatrices(float aspect) {
			aspectRatio = aspect;
			UpdateOrientation();
			viewMatrix = glm::lookAt(position, position + GetForward(), GetUp());
			projMatrix = glm::perspective(fov, aspectRatio, nearPlane, farPlane);
			projMatrix[1][1] *= -1.0f; // Vulkan inverted Y

			viewProjMatrix = projMatrix * viewMatrix;

			// Extract frustum planes from viewProjMatrix (Gribb-Hartmann method)
			const glm::mat4& m = viewProjMatrix;
			frustumPlanes[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0], m[2][3] + m[2][0], m[3][3] + m[3][0]); // Left
			frustumPlanes[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0], m[2][3] - m[2][0], m[3][3] - m[3][0]); // Right
			frustumPlanes[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1], m[2][3] + m[2][1], m[3][3] + m[3][1]); // Bottom
			frustumPlanes[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1], m[2][3] - m[2][1], m[3][3] - m[3][1]); // Top
			frustumPlanes[4] = glm::vec4(m[0][3] + m[0][2], m[1][3] + m[1][2], m[2][3] + m[2][2], m[3][3] + m[3][2]); // Near
			frustumPlanes[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2], m[2][3] - m[2][2], m[3][3] - m[3][2]); // Far

			for (int i = 0; i < 6; ++i) {
				float len = glm::length(glm::vec3(frustumPlanes[i]));
				if (len > 0.00001f) {
					frustumPlanes[i] /= len;
				}
			}
		}
	};

} // namespace brassica
