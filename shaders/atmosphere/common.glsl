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
