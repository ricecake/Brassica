#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct TonemapPushConstants {
		std::uint32_t hdrColorIndex{0};
		std::uint32_t bloomBlurIndex{0};
		std::uint32_t ltmFusedIndex{0};
		std::uint32_t ltmExpMipIndex{0};

		std::uint32_t depthTextureIndex{0};
		std::uint32_t toneMapMode{5};
		glm::vec2     ltmRes{0.0f, 0.0f};

		float intensity{0.075f};
		float minIntensity{0.05f};
		float maxIntensity{0.15f};
		float exposure{1.0f};

		float contrast{1.0f};
		float saturation{1.0f};
		float temperature{0.0f};
		float tint{0.0f};

		float pMax{1.0f};
		float pA{1.0f};
		float pM{0.22f};
		float pL{0.4f};

		float pC{1.33f};
		float pB{0.0f};
		float bloomIntensity{0.5f};
		float cdlSaturation{1.0f};

		glm::vec4 cdlSlope{1.0f, 1.0f, 1.0f, 1.0f};
		glm::vec4 cdlOffset{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4 cdlPower{1.0f, 1.0f, 1.0f, 1.0f};
	};

	inline TonemapPushConstants s_tonemapPush{};

} // namespace brassica
