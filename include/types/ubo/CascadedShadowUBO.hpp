#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct alignas(16) CascadedShadowUBO {
		glm::mat4 cascadeViewProj[4]{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)};
		glm::vec4 cascadeSplits{50.0f, 200.0f, 800.0f, 3200.0f}; // x = split0, y = split1, z = split2, w = split3
		glm::vec4 sunDirection{0.0f, 1.0f, 0.0f, 0.0f};         // xyz = light direction towards sun
		uint32_t  shadowMapIndex{0};                             // sampled bindless index into uTextureArrays[]
		uint32_t  shadowMapStorageIdx{0};                        // storage bindless index into uImageArraysRGBA32F[]
		float     shadowBias{0.0015f};
		uint32_t  numCascades{4};
	};

	static_assert(sizeof(CascadedShadowUBO) == 304, "CascadedShadowUBO struct size must be 304 bytes");

} // namespace brassica
