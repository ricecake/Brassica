#pragma once

#include <cstdint>

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
		// data that only changes when a UI slider moves, not per-draw. Mostly plain floats, which
		// (per std140) pack tightly after skyConvergenceStrength with no alignment padding
		// needed between them; the 3 vec3 color fields below are each alignas(16) so the
		// compiler inserts whatever padding std140 needs before them automatically, same as
		// rayleighScatteringBase/ozoneAbsorptionBase/hazeColor above.
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
		// Final flat tint on the cloud march's accumulated light (marchClouds()'s return value,
		// cloud_render.frag) -- boidish's cloudColorUniform applied the same way, as the very
		// last step before returning. Defaults to a no-op (1,1,1), not boidish's own tuned
		// {0.95,0.95,1.0} -- Brassica's extinction formula is its own adaptation, not boidish's
		// verbatim, so the port shouldn't silently change today's look; dial it in via the UI.
		alignas(16) glm::vec3 cloudColor{1.0f, 1.0f, 1.0f};
		// Per-channel extinction: rgbExtinction = cloudExtinctionColor * cloudExtinctionScale,
		// multiplied into the density->optical-depth->transmittance chain throughout
		// marchClouds() (stepDensity/stepOpticalDepth/cloudTransmittance, all vec3 for this).
		// cloudExtinctionScale defaults to 1.0 (no-op), not boidish's own 0.372, for the same
		// reason as cloudColor above.
		alignas(4) float cloudExtinctionScale{1.0f};
		alignas(16) glm::vec3 cloudExtinctionColor{1.0f, 1.0f, 1.0f};
		// Scales the cloud's reflected light (direct + ambient, NOT the lightning emissive
		// glow -- that's self-luminous, not reflective) before adding emissive back in. boidish
		// also multiplies by a per-sample albedo from calculateCloudDensity(), which Brassica's
		// live density function has no equivalent of, so this is the flat-uniform-only half of
		// that term. Defaults to a no-op (1,1,1), not boidish's own 0.85, for the same reason as
		// cloudColor above.
		alignas(16) glm::vec3 cloudAlbedo{1.0f, 1.0f, 1.0f};
		// GPU-driven lightning (LightningManager; shaders/climate_ca.comp's trigger,
		// shaders/lightning_age.comp's reserved-light-slot write) -- lightningFrequencyScale
		// multiplies the per-cell trigger probability; lightningIntensityScale scales bolt
		// brightness, the reserved uLights[] range's intensity, and cloud glow strength together.
		alignas(4) float lightningFrequencyScale{1.0f};
		alignas(4) float lightningIntensityScale{1.0f};
		// Sun aureole (shared by SkyBackgroundNode's sky.frag) -- previously hardcoded
		// push.sunDirAndAureole.w = 0.5f in SkyBackgroundNode::SetFrameParams, no UI at all.
		alignas(4) float sunAureoleStrength{0.5f};
		// Time-animated shimmer on sky_view_lut.comp's rayleigh/mie scattering coefficients --
		// previously had no GLSL consumer in Brassica at all.
		alignas(4) float colorVarianceScale{1.0f};
		alignas(4) float colorVarianceStrength{0.5f};

		// Volumetric fog settings (VolumetricLightingManager) -- same rationale, previously in
		// VolumetricFroxelGenPushConstants (fogParams.xyz plus mieHumidityBoost bit-cast through
		// a spare uint slot to avoid growing that struct past 128 bytes).
		alignas(4) float fogDensity{0.003f};
		alignas(4) float fogHeightFalloff{0.003f};
		alignas(4) float fogBaseHeight{0.0f};
		alignas(4) float mieHumidityBoost{5.0f};

		// More GPU-driven lightning settings (LightningManager) -- appended here rather than
		// next to lightningFrequencyScale/lightningIntensityScale above (push-constant-struct-
		// editing convention: always append new fields at the end, never insert mid-struct).
		// lightningSpawnRadius: climate_ca.comp only rolls a strike for cells within this many
		// metres of the camera's XZ position -- strikes used to spawn planet-wide, in cells the
		// camera could never see, which was most of why lightning read as "always firing"
		// (shaders/climate_ca.comp's trigger comment has the full rate derivation).
		alignas(4) float lightningSpawnRadius{60000.0f};
		// lightningStormThreshold: climate_ca.comp's stormFactor is
		// smoothstep(lightningStormThreshold, 1.0, lightningPropensity) -- how convective a cell
		// needs to be before it can trigger at all, on top of the base rate.
		alignas(4) float lightningStormThreshold{0.35f};
		// lightningCloudGlowRadius: metres. cloud_raymarch.comp's per-strike glow now falls off
		// with distance from the strike (exp(-(d/radius)^2)) instead of lighting every visible
		// cloud pixel on screen regardless of where the strike actually is.
		alignas(4) float lightningCloudGlowRadius{4000.0f};
		// lightningCloudFlashRadius: metres. The cloud-to-cloud FLASH light's radius
		// (lightning_age.comp) -- see LIGHT_TYPE_FLASH's innerCutoff-as-radius convention in
		// helpers/lighting.glsl.
		alignas(4) float lightningCloudFlashRadius{6000.0f};
		// lightningIgnoreWeather: debug override -- when nonzero, climate_ca.comp's stormFactor
		// is forced to 1.0 regardless of actual weather, so strikes can be seen without waiting
		// for a real storm. GPU equivalent of boidish's LightningWidget "Manual Auto-trigger".
		alignas(4) std::uint32_t lightningIgnoreWeather{0};
		// lightningGroundStrikeChance: fraction (0..1) of triggered strikes that become
		// cloud-to-ground instead of cloud-to-cloud (climate_ca.comp).
		alignas(4) float lightningGroundStrikeChance{0.3f};
		// lightningGroundFlashRadius: metres. The cloud-to-ground FLASH light's radius, placed
		// at the resolved ground point instead of the cloud-to-cloud midpoint -- bigger than
		// lightningCloudFlashRadius since a ground strike's visible illumination radius on
		// terrain is the whole point of it, not an incidental glow.
		alignas(4) float lightningGroundFlashRadius{1500.0f};
		// lightningLifetimeScale: multiplies climate_ca.comp's rolled strike duration -- boidish's
		// Lifetime Multiplier.
		alignas(4) float lightningLifetimeScale{1.0f};
		// Color variety (climate_ca.comp picks the final color at trigger time, not physically
		// modeled -- folk-meteorology color associations for artistic effect). Base is the
		// existing blue-white; humid/dusty are blend targets driven by that cell's own
		// airMoisture+rain and aerosol, respectively.
		alignas(16) glm::vec3 lightningBaseColor{0.8f, 0.85f, 1.0f};
		alignas(16) glm::vec3 lightningHumidColor{0.75f, 0.7f, 1.0f};
		alignas(16) glm::vec3 lightningDustyColor{1.0f, 0.85f, 0.6f};
		// 0..1. Per-strike brightness jitter strength on top of the weather-driven color above
		// (not a blend toward literal random noise -- that reads as broken rather than varied).
		alignas(4) float lightningColorVariance{0.15f};
	};

} // namespace brassica
