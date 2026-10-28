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

		// Cloud lighting settings (CloudLightingManager) -- previously threaded per-frame
		// through NodeFrameParams into CloudRenderPushConstants/CirrusPushConstants, which kept
		// pushing those structs toward the 128-byte guaranteed-safe push-constant ceiling for
		// data that only changes when a UI slider moves, not per-draw. All plain floats, so
		// (per std140) they pack tightly after skyConvergenceStrength with no alignment padding
		// needed between them -- no vec3/vec4 fields in this section.
		alignas(4) float cloudPhaseG1{0.8f};
		alignas(4) float cloudPhaseG2{-0.2f};
		alignas(4) float cloudPhaseAlpha{0.5f};
		alignas(4) float cloudPhaseIsotropic{0.52f};
		// ambientBeerPowderScale: Brassica's own ambient-glow term (cloud_render.frag). The
		// other 3 are beerPowder()'s own terms (shaders/common.glsl), previously hardcoded
		// literals -- not the same quantity as ambientBeerPowderScale despite boidish naming
		// its own equivalent of directBeerPowderScale "cloudPowderScale" too; see
		// CloudLightingManager.hpp's header comment.
		alignas(4) float ambientBeerPowderScale{300.0f};
		alignas(4) float directBeerPowderScale{0.1f};
		alignas(4) float cloudPowderMultiplier{50.0f};
		alignas(4) float cloudPowderLocalScale{2.67f};
		alignas(4) float cloudBeerPowderMix{0.5f};
		alignas(4) float cloudShadowOpticalDepthMultiplier{1.0f};
		alignas(4) float cloudSunLightScale{1.0f};
		alignas(4) float cloudMoonLightScale{1.0f};
		alignas(4) float cirrusOpacity{0.3f};
		alignas(4) float cloudFlowSpeed{0.25f};
		alignas(4) float cloudFlowDirection{3.14159265f};
		alignas(4) float cloudFlowHeightScale{0.015f};
		// Sun aureole (shared by SkyBackgroundNode's sky.frag) -- previously hardcoded
		// push.sunDirAndAureole.w = 0.5f in SkyBackgroundNode::SetFrameParams, no UI at all.
		alignas(4) float sunAureoleStrength{0.5f};

		// Volumetric fog settings (VolumetricLightingManager) -- same rationale, previously in
		// VolumetricFroxelGenPushConstants (fogParams.xyz plus mieHumidityBoost bit-cast through
		// a spare uint slot to avoid growing that struct past 128 bytes).
		alignas(4) float fogDensity{0.003f};
		alignas(4) float fogHeightFalloff{0.003f};
		alignas(4) float fogBaseHeight{0.0f};
		alignas(4) float mieHumidityBoost{5.0f};
	};

} // namespace brassica
