#pragma once

#include <glm/glm.hpp>

namespace brassica {

	struct SkyViewPushConstants {
		alignas(16) glm::vec3 sunDir{0.0f, 1.0f, 0.0f};
		alignas(4) float time{0.0f};
		alignas(16) glm::vec3 sunRadiance{3.0f, 2.94f, 2.76f};
		alignas(4) float worldScale{1.0f};
		alignas(16) glm::vec3 moonDir{0.0f, -1.0f, 0.0f};
		alignas(4) float multiScatScale{1.0f};
		alignas(16) glm::vec3 moonRadiance{0.1f, 0.12f, 0.16f};
		alignas(4) float cloudShadowIntensity{0.5f};
		alignas(16) glm::vec3 cameraPos{0.0f, 0.0f, 0.0f};
		alignas(4) float padding{0.0f};
	};

	static_assert(sizeof(SkyViewPushConstants) == 80, "SkyViewPushConstants size must be 80 bytes");

} // namespace brassica
