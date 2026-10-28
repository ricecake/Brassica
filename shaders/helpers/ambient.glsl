#ifndef HELPERS_AMBIENT_GLSL
#define HELPERS_AMBIENT_GLSL

// Temporary stand-in for a real spherical-harmonics ambient probe grid. LightingUBO already
// reserves uSHCoeffs[81] for exactly that (shaders/lighting.glsl), but nothing bakes it yet --
// these three functions approximate the same ground/sky/horizon split a probe grid would
// give, directly from the sun direction, so callers can be written against the same shape
// (three directional colors, attenuated by how much cloud is above vs below the sample) and
// swapped onto real probe data later without restructuring the calling code.

vec3 getApproximateSkyAmbient(vec3 sunDir, vec3 sunRadiance) {
	vec3  zenithColor = vec3(0.35, 0.55, 0.9);
	// Sky brightens with sun elevation (more scattered light overall) but never goes fully
	// dark at night -- a dim floor keeps night clouds from reading as pure black silhouettes.
	float skyBrightness = mix(0.08, 1.0, smoothstep(-0.1, 0.4, sunDir.y));
	return zenithColor * skyBrightness * max(sunRadiance.r, 0.05);
}

vec3 getApproximateGroundAmbient(vec3 sunRadiance) {
	vec3  groundColor = vec3(0.25, 0.35, 0.15);
	float groundBrightness = mix(0.05, 0.6, clamp(sunRadiance.r / 3.0, 0.0, 1.0));
	return groundColor * groundBrightness;
}

// Horizon color shifts warm (toward the sun's own color) when the sun is near the horizon --
// real Rayleigh scattering reddens light travelling a long path through the atmosphere -- and
// settles toward a neutral sky/ground blend when the sun is high overhead or well below.
vec3 getApproximateHorizonAmbient(vec3 sunDir, vec3 sunRadiance, vec3 skyAmbient, vec3 groundAmbient) {
	vec3  warmHorizon = vec3(1.0, 0.55, 0.3) * max(sunRadiance.r, 0.05);
	float horizonCloseness = 1.0 - smoothstep(0.0, 0.35, abs(sunDir.y));
	vec3  neutralHorizon = mix(groundAmbient, skyAmbient, 0.6);
	return mix(neutralHorizon, warmHorizon, horizonCloseness);
}

#endif // HELPERS_AMBIENT_GLSL
