#ifndef HELPERS_WIND_GLSL
#define HELPERS_WIND_GLSL

// Requires noise/trig_noise.glsl's cross_noise() already included by the caller (via
// common.glsl, which every current caller already includes before this) -- trig_noise.glsl
// has no include guard of its own, so this deliberately does not self-include it the way some
// other helpers/*.glsl files do for their (guarded) dependencies.
//
// Single shared source of "which way does the prevailing wind blow" for every system that
// needs it (CA weather/climate, cirrus advection, cloud volume/weather advection), replacing
// several independently-invented wind computations that used to disagree with each other.
//
// Returns a Hadley/Ferrel/polar three-cell atmospheric circulation vector: x = zonal
// (east-positive), y = meridional (north-positive), both in world-XZ-ish units (not
// normalized -- magnitude already encodes the cell's relative strength). `lat` is signed
// latitude in [-1, 1] (-1 = south pole, 0 = equator, 1 = north pole).
//
// Model (unchanged from climate_ca.comp's original per-cell version, just factored out so
// every caller uses the identical formula instead of each inventing its own): tropics
// (|lat| < 0.25) get easterly trade winds, mid-latitudes (0.25-0.6) get westerlies, polar
// regions (> 0.6) get easterlies again, blended smoothly across the boundaries.
vec2 getPrevailingWind(float lat) {
	float absLat = abs(lat);

	float hadleyWeight = 1.0 - smoothstep(0.25, 0.4, absLat);
	float ferrelWeight = smoothstep(0.25, 0.4, absLat) * (1.0 - smoothstep(0.6, 0.7, absLat));
	float polarWeight = smoothstep(0.6, 0.7, absLat);

	vec2 hadleyVec = vec2(-1.0, -sign(lat) * 0.5);
	vec2 ferrelVec = vec2(1.0, sign(lat) * 0.5);
	vec2 polarVec = vec2(-1.0, 0.0);

	return (hadleyVec * hadleyWeight) + (ferrelVec * ferrelWeight) + (polarVec * polarWeight);
}

// CA-specific: discretizes the continuous prevailing wind into an integer texel offset for
// per-cell upwind advection, with a curl-noise wobble so cell boundaries aren't perfectly
// straight lines. `worldPos` only needs x/z (used as the noise domain); `time` drives the
// wobble's slow drift.
//
// offsetDir is derived from the CONTINUOUS prevailingWind, not the rounded windDir: right at
// the Hadley/Ferrel (or Ferrel/polar) boundary, the two cells' smoothstep weights can land at
// 0.5/0.5, which cancels prevailingWind to exactly (0,0) -- rounding first and feeding that
// into cross()/normalize() would NaN there. Falling back to a fixed direction when the
// continuous vector is also near-zero keeps this well-defined at that exact latitude too.
ivec2 getDiscretizedWindOffset(vec2 prevailingWind, vec3 worldPos, float time) {
	ivec2 windDir = ivec2(round(clamp(prevailingWind, -1.0, 1.0)));

	vec3 wobble = clamp(cross_noise(vec3(worldPos.x, 0.0, worldPos.z) * 0.001, time * 0.01), 0.0, 1.0);

	vec3  windDir3D = vec3(prevailingWind.x, 0.0, prevailingWind.y);
	float windLen = length(windDir3D);
	vec3  offsetDir = windLen > 1e-4 ? normalize(cross(windDir3D, vec3(0.0, 1.0, 0.0))) : vec3(1.0, 0.0, 0.0);

	wobble = offsetDir * clamp(dot(wobble, offsetDir), 0.0, 1.0);
	windDir += ivec2(wobble.x, wobble.y);
	return clamp(windDir, -1, 1);
}

#endif // HELPERS_WIND_GLSL
