#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct alignas(16) LightingUBO {
		uint32_t numLights{0};          // offset 0
		float    dayTime{12.0f};        // offset 4
		float    nightFactor{0.0f};     // offset 8
		float    worldScale{1.0f};      // offset 12

		glm::vec4 ambientLight{0.1f, 0.1f, 0.1f, 1.0f};  // offset 16 (xyz = color, w = unused)
		glm::vec4 lightningColor{0.0f, 0.0f, 0.0f, 0.0f}; // offset 32 (xyz = color, w = pulse)

		float skyExposure{1.0f};       // offset 48
		float starExposure{1.0f};      // offset 52
		float terrainExposure{1.0f};   // offset 56
		float _paddingExposure{0.0f};  // offset 60

		glm::vec4 shCoeffs[81]{};      // offset 64 (81 * 16 = 1296 bytes)
	};

	static_assert(sizeof(LightingUBO) == 1360, "LightingUBO struct size must be 1360 bytes");

} // namespace brassica
