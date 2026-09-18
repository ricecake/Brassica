#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

#include "../common.glsl"

const float kEarthRadius = FAKE_PLANET_RADIUS / 1000.0; // 600.0 km

#ifdef ATMOSPHERE_NO_PUSH_CONSTANTS
const vec3 kRayleighScattering = vec3(5.802e-3, 13.558e-3, 33.100e-3);
const float kRayleighScaleHeight = 8.0;
const vec3 kOzoneAbsorption = vec3(0.650e-3, 1.881e-3, 0.085e-3);
const float kMieScaleHeight = 1.2;
const vec3 hazeColor = vec3(0.6, 0.7, 0.8);
const vec3 kMieScattering = vec3(3.996e-3);
const vec3 kMieExtinction = vec3(4.440e-3);
const float u_rayleighScale = 1.1;
const float u_mieScale = 0.35;
const float u_mieAnisotropy = 0.8;
const float kAtmosphereHeight = 100.0;
const float u_hazeDensity = 0.015;
const float u_hazeHeight = 20.0;
const float u_waterLevel = 0.0;
const vec3 kWaterScattering = vec3(0.003, 0.007, 0.012);
const float u_waterScale = 1.0;
const vec3 kWaterExtinction = vec3(0.12, 0.04, 0.02);
#else
#define kRayleighScattering u_atmosphere.rayleighScatteringBase
#define kRayleighScaleHeight u_atmosphere.rayleighScaleHeight
#define kOzoneAbsorption u_atmosphere.ozoneAbsorptionBase
#define kMieScaleHeight u_atmosphere.mieScaleHeight
#define hazeColor u_atmosphere.hazeColor
#define kMieScattering u_atmosphere.mieScatteringBase
#define kMieExtinction u_atmosphere.mieExtinctionBase
#define u_rayleighScale u_atmosphere.rayleighScale
#define u_mieScale u_atmosphere.mieScale
#define u_mieAnisotropy u_atmosphere.mieAnisotropy
#define kAtmosphereHeight u_atmosphere.atmosphereHeight
#define u_hazeDensity u_atmosphere.hazeDensity
#define u_hazeHeight u_atmosphere.hazeHeight
#define u_waterLevel u_atmosphere.waterLevel
#define kWaterScattering u_atmosphere.waterScatteringBase
#define u_waterScale u_atmosphere.waterScale
#define kWaterExtinction u_atmosphere.waterExtinctionBase
#endif
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
	vec3 rayleigh;
	vec3 mie;
	vec3 extinction;
	vec3 fogScattering;
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

	// Underwater scattering check: if h is below sea level / water level
	float waterLevelKM = u_waterLevel / 1000.0;
	if (h < waterLevelKM) {
		float depth = waterLevelKM - h;
		// Transition smoothly to water scattering & extinction coefficients
		vec3 wScat = kWaterScattering * u_waterScale;
		vec3 wExt = kWaterExtinction * u_waterScale;

		// Combine or override with water optical properties
		float wFactor = clamp(depth * 10.0, 0.0, 1.0); // smooth step transition across surface boundary
		s.rayleigh = mix(s.rayleigh, wScat, wFactor);
		s.mie = mix(s.mie, wScat * 0.5, wFactor);
		s.fogScattering = mix(s.fogScattering, wScat * 0.2, wFactor);
		s.extinction = mix(airExtinction, wExt, wFactor);
	} else {
		s.extinction = airExtinction;
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

/**
 * Scaffolding / Hook: Volumetric Lighting & Light Shafts
 *
 * Integration Point for Volumetric Light Shafts:
 * Future passes will evaluate cascaded shadow maps / shadow ray queries along view rays
 * to compute volumetric shadows, crepuscular rays, and localized light shafts.
 *
 * @param rayOrigin     - Start position of ray (world space, km or meters relative to camera)
 * @param rayDir        - Normalized ray direction
 * @param rayLength     - Length of ray segment
 * @param lightDir      - Sun/moon light direction vector
 * @param lightRadiance - Directional light radiance
 * @return In-scattered volumetric light contribution
 */
vec3 evaluateVolumetricLighting(vec3 rayOrigin, vec3 rayDir, float rayLength, vec3 lightDir, vec3 lightRadiance) {
	// SCAFFOLD: Currently evaluates unobstructed directional light in-scattering.
	// When volumetric shadow maps / ray queries are bound, sample visibility along the ray step.
	float cosTheta = dot(rayDir, lightDir);
	float phase = miePhase(cosTheta);
	float approxVisibility = 1.0; // Hook: replace with shadow map / ray query visibility
	return lightRadiance * phase * approxVisibility;
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

/**
 * Computes atmospheric and exponential fog aerial perspective for rendered surfaces over distance.
 * Takes camera position, ray direction, surface distance (km), light direction, and light radiance.
 * Returns in-scattered radiance and outputs transmittance.
 */
vec3 evaluateAerialPerspective(
	vec3 rayOrigin,
	vec3 rayDir,
	float distanceKM,
	vec3 sunDir,
	vec3 sunRadiance,
	out vec3 outTransmittance
) {
	const int kSteps = 16;
	float dt = distanceKM / float(kSteps);
	vec3 L = vec3(0.0);
	vec3 T = vec3(1.0);

	float cosTheta = dot(rayDir, sunDir);
	float pRayleigh = rayleighPhase(cosTheta);
	float pMie = miePhase(cosTheta);

	for (int i = 0; i < kSteps; ++i) {
		float t = (float(i) + 0.5) * dt;
		vec3 p = rayOrigin + rayDir * t;
		float h = p.y; // altitude in km relative to Y=0 sea level

		Sampling s = getAtmosphereProperties(h);

		// Volumetric lighting shadow hook (defaults to 1.0)
		vec3 vLight = evaluateVolumetricLighting(p, rayDir, dt, sunDir, sunRadiance);

		vec3 scatter = (s.rayleigh * pRayleigh + (s.mie + s.fogScattering) * pMie);
		vec3 inScattered = scatter * vLight * dt;

		L += T * inScattered;
		T *= exp(-s.extinction * dt);
	}

	outTransmittance = T;
	return L;
}

#endif
