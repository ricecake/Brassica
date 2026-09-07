#pragma once

#include <glm/glm.hpp>

namespace brassica {

	struct AtmospherePushConstants {
		alignas(16) glm::vec3 rayleighScatteringBase{5.802e-3f, 13.558e-3f, 33.100e-3f};
		alignas(4) float rayleighScaleHeight{8.0f};
		alignas(16) glm::vec3 ozoneAbsorptionBase{0.650e-3f, 1.881e-3f, 0.085e-3f};
		alignas(4) float mieScaleHeight{1.2f};
		alignas(16) glm::vec3 hazeColor{0.6f, 0.7f, 0.8f};
		alignas(4) float mieScatteringBase{3.996e-3f};
		alignas(4) float mieExtinctionBase{4.440e-3f};
		alignas(4) float rayleighScale{1.1f};
		alignas(4) float mieScale{0.35f};
		alignas(4) float mieAnisotropy{0.8f};
		alignas(4) float atmosphereHeight{100.0f};
		alignas(4) float hazeDensity{0.0f};
		alignas(4) float hazeHeight{20.0f};
	};

} // namespace brassica
