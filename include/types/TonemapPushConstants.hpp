#pragma once

#include <cstdint>

namespace brassica {

	struct TonemapPushConstants {
		std::uint32_t hdrColorIndex{0};
		float         exposure{1.0f};
		float         contrast{1.0f};
		float         saturation{1.0f};
		float         temperature{0.0f};
		float         tint{0.0f};
		float         pMax{1.0f};
		float         pA{1.0f};
		float         pM{0.22f};
		float         pL{0.4f};
		float         pC{1.33f};
		float         pB{0.0f};
	};

	inline TonemapPushConstants s_tonemapPush{};

} // namespace brassica
