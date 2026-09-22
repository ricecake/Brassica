#ifndef ATMOSPHERE_AERIAL_PERSPECTIVE_GLSL
#define ATMOSPHERE_AERIAL_PERSPECTIVE_GLSL

// Requires the caller to have already included "bindless.glsl" (for SAMPLE_LINEAR) before this
// file, same convention as sky_view_lut.comp/transmittance_lut.comp/multiscattering_lut.comp.
#include "common.glsl" // atmosphere/common.glsl

// Everything the atmosphere contributes over one view ray, in one value. Same contract as
// Material (shaders/helpers/material.glsl): adding a term later means adding a field here with a
// default in atmosphereSampleDefault() and consuming it inside applyAtmosphere() -- every call
// site keeps passing the struct through unchanged.
//
//  cloudRadiance/cloudCoverage: reserved for the raymarched-cloud path. Zero today; the hook is
//    evaluateRaymarchedClouds (common.glsl). Clouds are deliberately NOT folded into inScattering,
//    because they're eventually a separate low-res, temporally-accumulated pass composited via
//    combineAtmosphere, not part of this per-pixel march.
//  opticalDepth: -log of the mean transmittance. Free (already computed) and it is exactly the
//    driver a future distance-blur term needs, so it's recorded now rather than re-derived later.
struct AtmosphereSample {
	vec3  inScattering;
	vec3  transmittance;
	vec3  cloudRadiance;
	float cloudCoverage;
	float opticalDepth;
};

AtmosphereSample atmosphereSampleDefault() {
	return AtmosphereSample(vec3(0.0), vec3(1.0), vec3(0.0), 0.0, 0.0);
}

// Everything evaluateAtmosphere needs about one ray. An input struct for the same reason as the
// output one: volumetric shadows will need a shadow-map index, clouds a weather-texture index,
// froxel volumetrics a 3D-volume index -- each is a field here plus one line at the call site.
// Tuning constants (kRayleighScattering etc.) come from the global u_atmosphere UBO directly
// inside marchAtmosphereSegment, same as every other atmosphere shader -- not threaded through
// this struct.
struct AtmosphereRay {
	vec3  originKM;    // planet-centric: vec3(0, kEarthRadius + camAltKM, 0) -- see sky_view_lut.comp
	vec3  direction;   // normalized, world space (flat-world approximation, same as sky_view_lut.comp)
	float distanceKM;  // path length to shade over
	vec3  sunDir;
	vec3  sunRadiance;
	vec3  moonDir;
	vec3  moonRadiance;
	float multiScatScale;
	uint  transmittanceIndex;
	uint  multiScatteringIndex;
	uint  steps;
};

// Replaces evaluateVolumetricLighting: returns unitless per-channel visibility (not radiance --
// the old signature returned lightRadiance * phase * visibility, which double-applied the phase
// function once marchAtmosphereSegment separately multiplied by the same phase again, and got
// multiplied straight into the lit surface color by whoever called it instead of added). Hook:
// replace vec3(1.0) with a shadow-map / ray-query lookup at posKM.
vec3 evaluateVolumetricVisibility(vec3 posKM, vec3 lightDir) {
	return vec3(1.0);
}

vec3 sampleTransmittance(uint index, float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(index, uv).rgb;
}

vec3 sampleMultiScattering(uint index, float r, float mu_s) {
	vec2 uv = vec2(mu_s * 0.5 + 0.5, (r - kEarthRadius) / kAtmosphereHeight);
	return SAMPLE_LINEAR(index, uv).rgb;
}

// Marches [t0KM, t1KM] along ray only -- evaluateAtmosphere composes two of these when the ray
// crosses the water plane. A future AerialPerspectiveLUTNode's compute shader calls this directly
// so the LUT and the reference path can never disagree. Body is sky_view_lut.comp's existing
// sun+moon in-scattering integral (lines ~69-96 there), generalized to an arbitrary sub-segment
// and an explicit step count, plus two real fixes: s.fogScattering (near-ground haze, the
// dominant term for "distant mountains swallowed") folded into the Mie lobe, which sky_view_lut's
// own sky-only integral doesn't need but ground-level aerial perspective does; and
// evaluateVolumetricVisibility multiplying the direct term instead of the old
// evaluateAerialPerspective's double-phase-application bug.
AtmosphereSample marchAtmosphereSegment(AtmosphereRay ray, float t0KM, float t1KM, uint steps) {
	AtmosphereSample result = atmosphereSampleDefault();

	float segmentLength = t1KM - t0KM;
	if (segmentLength <= 0.0 || steps == 0u) {
		return result;
	}

	float dt = segmentLength / float(steps);
	vec3  L = vec3(0.0);
	vec3  T = vec3(1.0);

	float cosThetaSun = dot(ray.direction, ray.sunDir);
	float cosThetaMoon = dot(ray.direction, ray.moonDir);
	bool  hasMoon = length(ray.moonRadiance) > 0.00001;

	for (uint i = 0u; i < steps; ++i) {
		float t = t0KM + (float(i) + 0.5) * dt;
		vec3  p = ray.originKM + ray.direction * t;
		float r = length(p);
		float h = r - kEarthRadius;

		Sampling s = getAtmosphereProperties(h);

		// The multi-scattering LUT is precomputed for air's scattering coefficients (rayleigh
		// ~5-33e-3, mie ~4e-3) -- three orders of magnitude smaller than water's (~15-43, see
		// AtmospherePushConstants.hpp). Its gain factor has no physical validity once the sample is
		// substantially water, so fade it out by (1 - waterBlend) instead of reusing it unchanged
		// against a medium it was never computed for. Confirmed by hand: without this, underwater
		// in-scattering came out 10-100x brighter than the sun itself.
		float multiScatAirWeight = 1.0 - s.waterBlend;

		float cosSun = dot(normalize(p), ray.sunDir);
		vec3  Ts = sampleTransmittance(ray.transmittanceIndex, r, cosSun);
		vec3  multiScatSun = sampleMultiScattering(ray.multiScatteringIndex, r, cosSun) * ray.multiScatScale * multiScatAirWeight;
		vec3  phaseSun = s.rayleigh * rayleighPhase(cosThetaSun) + (s.mie + s.fogScattering) * miePhase(cosThetaSun);
		vec3  visSun = evaluateVolumetricVisibility(p, ray.sunDir);

		vec3 source = (phaseSun * Ts * visSun + multiScatSun * (s.rayleigh + s.mie)) * ray.sunRadiance;

		if (hasMoon) {
			float cosMoon = dot(normalize(p), ray.moonDir);
			vec3  Tm = sampleTransmittance(ray.transmittanceIndex, r, cosMoon);
			vec3  multiScatMoon =
				sampleMultiScattering(ray.multiScatteringIndex, r, cosMoon) * ray.multiScatScale * multiScatAirWeight;
			vec3 phaseMoon =
				s.rayleigh * rayleighPhase(cosThetaMoon) + (s.mie + s.fogScattering) * miePhase(cosThetaMoon);
			vec3 visMoon = evaluateVolumetricVisibility(p, ray.moonDir);

			source += (phaseMoon * Tm * visMoon + multiScatMoon * (s.rayleigh + s.mie)) * ray.moonRadiance;
		}

		// Analytic per-step integral (assumes source/extinction constant across the step, exact in
		// that limit), not a naive Riemann sum (source * dt): the two agree when extinction * dt is
		// small (true for the existing thin-atmosphere case this loop body is adapted from,
		// sky_view_lut.comp), but the naive sum badly *overestimates* in-scattering once a step's
		// optical depth isn't small -- exactly the water case here (water's real extinction, ~20-
		// 280/km, times a several-metre step is nowhere near "small"). Confirmed by hand: this is
		// what fixed underwater in-scattering values that were 10-100x brighter than the sun itself.
		vec3 stepTransmittance = exp(-s.extinction * dt);
		vec3 integralWeight = (vec3(1.0) - stepTransmittance) / max(s.extinction, vec3(1e-6));

		L += T * source * integralWeight;
		T *= stepTransmittance;
	}

	result.inScattering = L;
	result.transmittance = T;
	result.opticalDepth = -log(max(1e-6, dot(T, vec3(0.2126, 0.7152, 0.0722))));
	return result;
}

// Associative composition: `near` is the camera-side segment. This is the operator that lets
// clouds and volumetrics arrive as separate passes/segments and still combine correctly without
// any consumer knowing how many terms there are.
AtmosphereSample combineAtmosphere(AtmosphereSample near, AtmosphereSample far) {
	AtmosphereSample r;
	r.inScattering = near.inScattering + near.transmittance * far.inScattering;
	r.transmittance = near.transmittance * far.transmittance;
	r.cloudRadiance = near.cloudRadiance + near.transmittance * far.cloudRadiance;
	r.cloudCoverage = near.cloudCoverage + (1.0 - near.cloudCoverage) * far.cloudCoverage;
	r.opticalDepth = near.opticalDepth + far.opticalDepth;
	return r;
}

// The one choke point: raymarch today, a LUT fetch later, with no change to any call site.
// Splits the march at the water-plane crossing when the ray passes through it, so a submerged
// camera looking up at a mountain doesn't under-integrate the water column by only getting a
// couple of the total steps inside it.
AtmosphereSample evaluateAtmosphere(AtmosphereRay ray) {
	float waterLevelKM = u_waterLevel / 1000.0;
	float camAltKM = ray.originKM.y - kEarthRadius;

	bool  crossesWater = abs(ray.direction.y) > 1e-6;
	float tCrossKM = crossesWater ? (waterLevelKM - camAltKM) / ray.direction.y : -1.0;

	if (crossesWater && tCrossKM > 1e-6 && tCrossKM < ray.distanceKM) {
		uint stepsNear = max(8u, ray.steps / 2u);
		uint stepsFar = max(8u, ray.steps - stepsNear);
		AtmosphereSample nearSeg = marchAtmosphereSegment(ray, 0.0, tCrossKM, stepsNear);
		AtmosphereSample farSeg = marchAtmosphereSegment(ray, tCrossKM, ray.distanceKM, stepsFar);
		return combineAtmosphere(nearSeg, farSeg);
	}

	return marchAtmosphereSegment(ray, 0.0, ray.distanceKM, max(1u, ray.steps));
}

// The one place scene radiance and atmosphere meet.
vec3 applyAtmosphere(AtmosphereSample s, vec3 sceneRadiance) {
	vec3 fogged = sceneRadiance * s.transmittance + s.inScattering;
	return mix(fogged, s.cloudRadiance, s.cloudCoverage);
}

#endif // ATMOSPHERE_AERIAL_PERSPECTIVE_GLSL
