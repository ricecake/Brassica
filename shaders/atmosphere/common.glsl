#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

#include "../common.glsl"
#include "../bindless.glsl"

const float kEarthRadius = FAKE_PLANET_RADIUS / 1000.0; // 600.0 km

// Real, globally-bound tuning data -- not a per-node push constant, so any shader that includes
// this file gets it for free, and it's the hook for editing these values via an interface later
// without touching a single call site. Mirrors AtmospherePushConstants (types/AtmospherePushConstants.hpp)
// field-for-field; keep the two in sync by hand, same convention as every other *PushConstants
// mirror in this codebase.
layout(std140, set = 0, binding = 4) uniform AtmosphereUBO {
	vec3  rayleighScatteringBase;
	float rayleighScaleHeight;
	vec3  ozoneAbsorptionBase;
	float mieScaleHeight;
	vec3  hazeColor;
	float mieScatteringBase;
	float mieExtinctionBase;
	float rayleighScale;
	float mieScale;
	float mieAnisotropy;
	float atmosphereHeight;
	float hazeDensity;
	float hazeHeight;
	float waterLevel;
	float waterBlendDepth;
	vec3  waterScatteringBase;
	float waterScale;
	vec3  waterExtinctionBase;
	// 0..1. How strongly aerial_perspective.glsl's sky-color convergence pulls distant fogged
	// geometry toward the real SkyViewLUT radiance for that view direction, weighted by fog
	// opacity -- see marchAtmosphereSegment's caller in composite.frag. 1.0 = fully physically
	// motivated blend; 0.0 = old behavior (pure single/multi-scatter march, no convergence).
	float skyConvergenceStrength;

	// Cloud lighting settings (CloudLightingManager) -- previously threaded per-frame through
	// NodeFrameParams into CloudRenderPushConstants/CirrusPushConstants, which kept pushing
	// those structs toward the 128-byte guaranteed-safe push-constant ceiling for data that
	// only changes when a UI slider moves, not per-draw.
	float cloudPhaseG1;
	float cloudPhaseG2;
	float cloudPhaseAlpha;
	float cloudPhaseIsotropic;
	// ambientBeerPowderScale: Brassica's own ambient-glow term (cloud_render.frag). The other
	// 3 are beerPowder()'s own terms (shaders/common.glsl's beerPowder function), previously
	// hardcoded literals -- not the same quantity as ambientBeerPowderScale.
	float ambientBeerPowderScale;
	float directBeerPowderScale;
	float cloudPowderMultiplier;
	float cloudPowderLocalScale;
	float cloudBeerPowderMix;
	float cloudShadowOpticalDepthMultiplier;
	float cloudSunLightScale;
	float cloudMoonLightScale;
	float cirrusOpacity;
	float cloudFlowSpeed;
	float cloudFlowDirection;
	float cloudFlowHeightScale;
	// Final flat tint on the cloud march's accumulated light -- applied as the very last step
	// before marchClouds() returns (cloud_render.frag).
	vec3  cloudColor;
	// Per-channel extinction: rgbExtinction = cloudExtinctionColor * cloudExtinctionScale,
	// multiplied into the density->optical-depth->transmittance chain throughout
	// marchClouds() (stepDensity/stepOpticalDepth/cloudTransmittance, all vec3 for this).
	float cloudExtinctionScale;
	vec3  cloudExtinctionColor;
	// Scales the cloud's reflected light (direct + ambient, NOT the lightning emissive glow).
	vec3  cloudAlbedo;
	// GPU-driven lightning (climate_ca.comp's trigger, lightning_age.comp's reserved-light-
	// slot write) -- lightningFrequencyScale multiplies the per-cell trigger probability;
	// lightningFlashIntensity scales the reserved uLights[] range's intensity.
	float lightningFrequencyScale;
	float lightningFlashIntensity;
	// Sun aureole (SkyBackgroundNode's sky.frag) -- previously a hardcoded 0.5f push-constant
	// literal with no settings surface.
	float sunAureoleStrength;
	// Time-animated shimmer on sky_view_lut.comp's rayleigh/mie scattering coefficients --
	// previously had no GLSL consumer in Brassica at all.
	float colorVarianceScale;
	float colorVarianceStrength;

	// Volumetric fog settings (VolumetricLightingManager) -- same rationale, previously in
	// VolumetricFroxelGenPushConstants (fogParams.xyz plus mieHumidityBoost bit-cast through a
	// spare uint slot to avoid growing that struct past 128 bytes).
	float fogDensity;
	float fogHeightFalloff;
	float fogBaseHeight;
	float mieHumidityBoost;
}
u_atmosphere;

#define kRayleighScattering u_atmosphere.rayleighScatteringBase
#define kRayleighScaleHeight u_atmosphere.rayleighScaleHeight
#define kOzoneAbsorption u_atmosphere.ozoneAbsorptionBase
#define kMieScaleHeight u_atmosphere.mieScaleHeight
#define hazeColor u_atmosphere.hazeColor
// mieScatteringBase/mieExtinctionBase are scalars (uniform across wavelengths) on the C++ side;
// every consumer here wants a vec3, so broadcast explicitly at the alias site rather than at every
// use site.
#define kMieScattering vec3(u_atmosphere.mieScatteringBase)
#define kMieExtinction vec3(u_atmosphere.mieExtinctionBase)
#define u_rayleighScale u_atmosphere.rayleighScale
#define u_mieScale u_atmosphere.mieScale
#define u_mieAnisotropy u_atmosphere.mieAnisotropy
#define kAtmosphereHeight u_atmosphere.atmosphereHeight
#define u_hazeDensity u_atmosphere.hazeDensity
#define u_hazeHeight u_atmosphere.hazeHeight
#define u_waterLevel u_atmosphere.waterLevel
#define u_waterBlendDepth u_atmosphere.waterBlendDepth
#define kWaterScattering u_atmosphere.waterScatteringBase
#define u_waterScale u_atmosphere.waterScale
#define kWaterExtinction u_atmosphere.waterExtinctionBase
#define u_skyConvergence u_atmosphere.skyConvergenceStrength
#define kTopRadius (kEarthRadius + kAtmosphereHeight)

// Cloud/volumetric settings aliases -- deliberately `u_`-prefixed, NOT bare field names: the
// bare names (fogDensity, mieHumidityBoost, ...) still collide with existing push-constant
// struct members/locals of the exact same name in volumetric_froxel_gen.comp (Stage C,
// pending), which hasn't been migrated off push constants onto this UBO yet. cloud_render.frag
// and cirrus.frag have already been migrated (CloudRenderPushConstants/CirrusPushConstants no
// longer declare these fields) but the aliases stay `u_`-prefixed regardless, so this section
// doesn't need touching again once volumetric_froxel_gen.comp's migration lands.
#define u_cloudPhaseG1 u_atmosphere.cloudPhaseG1
#define u_cloudPhaseG2 u_atmosphere.cloudPhaseG2
#define u_cloudPhaseAlpha u_atmosphere.cloudPhaseAlpha
#define u_cloudPhaseIsotropic u_atmosphere.cloudPhaseIsotropic
#define u_ambientBeerPowderScale u_atmosphere.ambientBeerPowderScale
#define u_directBeerPowderScale u_atmosphere.directBeerPowderScale
#define u_cloudPowderMultiplier u_atmosphere.cloudPowderMultiplier
#define u_cloudPowderLocalScale u_atmosphere.cloudPowderLocalScale
#define u_cloudBeerPowderMix u_atmosphere.cloudBeerPowderMix
#define u_cloudShadowOpticalDepthMultiplier u_atmosphere.cloudShadowOpticalDepthMultiplier
#define u_cloudSunLightScale u_atmosphere.cloudSunLightScale
#define u_cloudMoonLightScale u_atmosphere.cloudMoonLightScale
#define u_cirrusOpacity u_atmosphere.cirrusOpacity
#define u_cloudFlowSpeed u_atmosphere.cloudFlowSpeed
#define u_cloudFlowDirection u_atmosphere.cloudFlowDirection
#define u_cloudFlowHeightScale u_atmosphere.cloudFlowHeightScale
#define u_cloudColor u_atmosphere.cloudColor
#define u_cloudExtinctionScale u_atmosphere.cloudExtinctionScale
#define u_cloudExtinctionColor u_atmosphere.cloudExtinctionColor
#define u_cloudAlbedo u_atmosphere.cloudAlbedo
#define u_lightningFrequencyScale u_atmosphere.lightningFrequencyScale
#define u_lightningFlashIntensity u_atmosphere.lightningFlashIntensity
#define u_sunAureoleStrength u_atmosphere.sunAureoleStrength
#define u_colorVarianceScale u_atmosphere.colorVarianceScale
#define u_colorVarianceStrength u_atmosphere.colorVarianceStrength
#define u_fogDensity u_atmosphere.fogDensity
#define u_fogHeightFalloff u_atmosphere.fogHeightFalloff
#define u_fogBaseHeight u_atmosphere.fogBaseHeight
#define u_mieHumidityBoost u_atmosphere.mieHumidityBoost

bool intersectSphere(vec3 ro, vec3 rd, float radius, out float t0, out float t1) {
	float b = dot(ro, rd);
	float rLen = length(ro);
	float c = (rLen - radius) * (rLen + radius);
	float det = b * b - c;
	if (det < 0.0)
		return false;
	det = sqrt(det);
	t0 = -b - det;
	t1 = -b + det;
	return true;
}

bool intersectCloudShell(vec3 ro, vec3 rd, float radius, float height, float thickness, out float t_start1, out float t_end1, out float t_start2, out float t_end2) {
	float R_earth = radius;
	float R_floor = R_earth + height;
	float R_ceiling = R_floor + thickness;

	vec3 earthCenter = vec3(uCameraPosition.x, -R_earth, uCameraPosition.z);
	vec3 relRo = ro - earthCenter;

	t_start1 = 0.0;
	t_end1 = -1.0;
	t_start2 = 0.0;
	t_end2 = -1.0;

	float t_out0, t_out1;
	if (!intersectSphere(relRo, rd, R_ceiling, t_out0, t_out1) || t_out1 <= 0.0) {
		return false;
	}

	// An intersection with the Earth sphere stops the ray
	float t_e0, t_e1;
	bool hits_earth = intersectSphere(relRo, rd, R_earth, t_e0, t_e1) && (t_e0 > 0.0);
	float t_max = hits_earth ? t_e0 : t_out1;

	float t_c_entry = max(0.0, t_out0);
	float t_c_exit  = min(t_out1, t_max);

	if (t_c_entry >= t_c_exit) {
		return false;
	}

	float t_in0, t_in1;
	if (intersectSphere(relRo, rd, R_floor, t_in0, t_in1)) {
		if (t_in0 < 0.0) {
			// Ray origin is inside R_floor (e.g., ground looking up into the sky)
			float ts = max(t_c_entry, t_in1);
			float te = t_c_exit;
			if (ts < te) {
				t_start1 = ts;
				t_end1 = te;
				return true;
			}
			return false;
		} else {
			// Ray enters R_floor at t_in0 > 0.0
			// Segment 1: from entry into atmosphere ceiling up to entering R_floor
			float ts1 = t_c_entry;
			float te1 = min(t_c_exit, t_in0);
			if (ts1 < te1) {
				t_start1 = ts1;
				t_end1 = te1;
			}

			// Segment 2: from exiting R_floor at t_in1 up to exiting R_ceiling (if not blocked by Earth)
			float ts2 = max(t_c_entry, t_in1);
			float te2 = t_c_exit;
			if (ts2 < te2) {
				if (t_start1 >= t_end1) {
					t_start1 = ts2;
					t_end1 = te2;
				} else {
					t_start2 = ts2;
					t_end2 = te2;
				}
			}

			return (t_start1 < t_end1) || (t_start2 < t_end2);
		}
	} else {
		// Does not intersect R_floor (grazes cloud layer without reaching floor)
		t_start1 = t_c_entry;
		t_end1 = t_c_exit;
		return t_start1 < t_end1;
	}
}

bool intersectCloudShell(vec3 ro, vec3 rd, float radius, float height, float thickness, out float t_start, out float t_end) {
	float t_s1, t_e1, t_s2, t_e2;
	if (intersectCloudShell(ro, rd, radius, height, thickness, t_s1, t_e1, t_s2, t_e2)) {
		if (t_s1 < t_e1) {
			t_start = t_s1;
			t_end = t_e1;
			return true;
		} else if (t_s2 < t_e2) {
			t_start = t_s2;
			t_end = t_e2;
			return true;
		}
	}
	t_start = 1e10;
	t_end = -1e10;
	return false;
}


float getRayleighDensity(float h) {
	return exp(-max(0.0, h) / kRayleighScaleHeight);
}

float getMieDensity(float h) {
	return exp(-max(0.0, h) / kMieScaleHeight);
}

float getOzoneDensity(float h) {
	return max(0.0, 1.0 - abs(max(0.0, h) - 25.0) / 15.0);
}

float getExponentialFogDensity(float h) {
	return exp(-max(0.0, h) / max(0.001, u_hazeHeight)) * u_hazeDensity;
}

struct Sampling {
	vec3  rayleigh;
	vec3  mie;
	vec3  extinction;
	vec3  fogScattering;
	// 0 = pure air, 1 = fully water. Exposed so a multi-scattering LUT precomputed for air's
	// (three-orders-of-magnitude smaller) scattering coefficients can fade its own contribution
	// out rather than being reused unchanged against water's -- see aerial_perspective.glsl's
	// marchAtmosphereSegment.
	float waterBlend;
};

Sampling getAtmosphereProperties(float h) {
	float rd = getRayleighDensity(h);
	float md = getMieDensity(h);
	float od = getOzoneDensity(h);
	float fd = getExponentialFogDensity(h);

	Sampling s;
	s.rayleigh = kRayleighScattering * rd * u_rayleighScale;
	s.mie = hazeColor * (kMieScattering * md * u_mieScale);
	s.fogScattering = hazeColor * fd;
	vec3 airExtinction = s.rayleigh + hazeColor * (kMieExtinction * md * u_mieScale) + kOzoneAbsorption * od + s.fogScattering;

	// Underwater: if h is below sea level, transition smoothly to water scattering/extinction.
	// wFactor's width (u_waterBlendDepth, metres) used to be a hardcoded depth*10.0, saturating at
	// 100m -- a camera 2m underwater got ~98% air optics regardless of the coefficients below.
	float waterLevelKM = u_waterLevel / 1000.0;
	if (h < waterLevelKM) {
		float depth = waterLevelKM - h;
		float blendKM = max(1e-6, u_waterBlendDepth / 1000.0);
		float wFactor = clamp(depth / blendKM, 0.0, 1.0);
		vec3  wScat = kWaterScattering * u_waterScale;
		vec3  wExt = kWaterExtinction * u_waterScale;

		// All water scattering routed through the single forward (Mie-phase) lobe, total exactly
		// wScat -- summing scattering across rayleigh+mie+fog terms independently would break the
		// scattering/extinction == shallowWaterTint identity waterScatteringBase's default is
		// derived from (see AtmospherePushConstants.hpp).
		s.rayleigh = mix(s.rayleigh, vec3(0.0), wFactor);
		s.mie = mix(s.mie, wScat, wFactor);
		s.fogScattering = mix(s.fogScattering, vec3(0.0), wFactor);
		s.extinction = mix(airExtinction, wExt, wFactor);
		s.waterBlend = wFactor;
	} else {
		s.extinction = airExtinction;
		s.waterBlend = 0.0;
	}

	return s;
}

float rayleighPhase(float cosTheta) {
	return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float miePhase(float cosTheta) {
	float g = u_mieAnisotropy;
	float g2 = g * g;
	return (1.0 - g2) / (4.0 * PI * pow(max(1e-4, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

vec2 transmittanceToUV(float r, float mu) {
	float x_mu = mu * 0.5 + 0.5;
	float x_r = (r - kEarthRadius) / kAtmosphereHeight;
	return vec2(x_mu, x_r);
}

void UVToTransmittance(vec2 uv, out float r, out float mu) {
	mu = uv.x * 2.0 - 1.0;
	r = kEarthRadius + uv.y * kAtmosphereHeight;
}

// Inverts sky_view_lut.comp's own elevation/azimuth parameterization of its output texture.
// Shared by SkyBackgroundNode's full-screen sky draw and aerial_perspective.glsl's sky-color
// convergence -- both need "what does the sky look like along this world-space ray," just for
// different pixels (empty-background vs. real fogged geometry).
vec3 sampleSkyView(uint index, vec3 rd) {
	float elevation = asin(clamp(rd.y, -1.0, 1.0));
	float azimuth = atan(rd.x, -rd.z);
	if (azimuth < 0.0)
		azimuth += 2.0 * PI;

	float v = (elevation < 0.0) ? (0.5 - 0.5 * sqrt(-elevation / (PI * 0.5)))
								: (0.5 + 0.5 * sqrt(elevation / (PI * 0.5)));
	vec2 uv = vec2(azimuth / (2.0 * PI), v);
	return SAMPLE_LINEAR(index, uv).rgb;
}

/**
 * Scaffolding / Hook: Raymarched Volumetric Clouds
 *
 * Integration Point for Raymarched Clouds:
 * Future passes will sample 3D noise weather maps (Perlin-Worley / erosion noise) between cloud
 * base altitude (~1.5 km) and cloud top altitude (~4.0 km) to render volumetric cloud layers with
 * multiple scattering and shadow mapping.
 *
 * @param rayOrigin   - World ray origin (km)
 * @param rayDir      - Normalized world ray direction
 * @param maxDistance - Maximum ray distance (km)
 * @param sunDir      - Direction to sun
 * @param sunRadiance - Direct sun radiance
 * @return vec4(rgb = cloud radiance, a = cloud transmittance / opacity)
 */
vec4 evaluateRaymarchedClouds(vec3 rayOrigin, vec3 rayDir, float maxDistance, vec3 sunDir, vec3 sunRadiance) {
	// SCAFFOLD: Default returns transparent clouds (0 opacity).
	// Future implementation will march between cloud bottom (~1.5km) and cloud top (~4.0km)
	// sampling weather textures, calculating powder effect & Henyey-Greenstein phase scattering.
	return vec4(0.0, 0.0, 0.0, 0.0);
}

#endif
