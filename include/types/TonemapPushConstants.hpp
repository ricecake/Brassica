#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct TonemapPushConstants {
		std::uint32_t hdrColorIndex{0};
		std::uint32_t bloomTextureIndex{0};
		std::uint32_t toneMapMode{
			5
		}; // 0=ACES, 1=Filmic, 2=Lottes, 3=Reinhard, 4=Reinhard2, 5=Uchimura, 6=Uncharted2, 7=Unreal, 8=Debug, 9=None
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
