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

		float time{0.0f};
		float fov{1.2f};
		float aspectRatio{16.0f / 9.0f};
		float nearPlane{0.1f};

		float    farPlane{32768.0f};
		uint32_t frameIndex{0};
		uint32_t globalSeed{0};
		uint32_t frameRandom{0};
	};

	static_assert(sizeof(FrameUBO) == 432, "FrameUBO struct size must be 432 bytes");

} // namespace brassica
