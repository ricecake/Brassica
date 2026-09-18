#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct SkyPushConstants {
		alignas(16) glm::vec4 sunDirAndAureole{0.0f, 1.0f, 0.0f, 0.5f};
		alignas(16) glm::vec4 moonDirAndCirrus{0.0f, -1.0f, 0.0f, 0.3f};
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f};
		alignas(4) float worldScale{1.0f};
		alignas(4) std::uint32_t skyViewIndex{0};
		alignas(4) std::uint32_t transmittanceIndex{0};
		alignas(4) float padding{0.0f};
	};

	static_assert(sizeof(SkyPushConstants) == 64, "SkyPushConstants size must be 64 bytes");

} // namespace brassica
