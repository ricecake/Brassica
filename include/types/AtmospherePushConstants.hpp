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
		alignas(4) float hazeDensity{0.015f};
		alignas(4) float hazeHeight{20.0f};
		alignas(4) float waterLevel{0.0f};
		// Metres. The air<->water optical transition width (atmosphere/common.glsl's
		// getAtmosphereProperties): replaces a hardcoded `depth * 10.0` that saturated at 100m,
		// making shallow submersion (1-2m) read as ~98% air optics regardless of the coefficients
		// below.
		alignas(4) float waterBlendDepth{2.0f};
		// Per kilometre (matches the km-based aerial-perspective raymarch integration, NOT
		// water.frag's own per-metre extinctionCoeff -- converting between the two is exactly
		// {280,70,20} vs {0.28,0.07,0.02}). Derived so scattering/extinction (the single-scatter
		// albedo) equals water.frag's shallowWaterTint = {0.12, 0.62, 0.78}, so this and water.frag's
		// already-correct surface math agree on what water looks like.
		alignas(16) glm::vec3 waterScatteringBase{33.6f, 43.4f, 15.6f};
		alignas(4) float waterScale{1.0f};
		alignas(16) glm::vec3 waterExtinctionBase{280.0f, 70.0f, 20.0f};
		// 0..1. How strongly the aerial-perspective composite pulls distant fogged geometry toward
		// the real SkyViewLUT radiance for that view direction (shaders/atmosphere/composite.frag),
		// weighted by fog opacity. 1.0 = fully physically motivated blend.
		alignas(4) float skyConvergenceStrength{1.0f};
	};

} // namespace brassica
