#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct alignas(16) FrameUBO {
		glm::mat4 viewMatrix{1.0f};
		glm::mat4 invViewMatrix{1.0f};
		glm::mat4 projMatrix{1.0f};
		glm::mat4 invProjMatrix{1.0f};
		glm::mat4 viewProjMatrix{1.0f};
		glm::mat4 invViewProjMatrix{1.0f};

		glm::vec4 cameraPosition{0.0f, 15.0f, 30.0f, 1.0f}; // xyz = position, w = baseTexelSize
		glm::vec4 previousCameraPosition{-999.0f, -15.0f, -30.0f, 0.0f};
		float time{0.0f};
		float fov{1.2f};
		float aspectRatio{16.0f / 9.0f};
		float nearPlane{0.1f};

		float    farPlane{32768.0f};
		uint32_t frameIndex{0};
		uint32_t globalSeed{0};
		uint32_t frameRandom{0};
	};

	static_assert(sizeof(FrameUBO) == 448, "FrameUBO struct size must be 448 bytes");

} // namespace brassica
