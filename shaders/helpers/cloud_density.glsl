#ifndef HELPERS_CLOUD_DENSITY_GLSL
#define HELPERS_CLOUD_DENSITY_GLSL

#include "bindless.glsl"
#include "common.glsl"
#include "atmosphere/common.glsl"
#include "helpers/constants.glsl"

// Normalized [0,1] fraction through the cloud band (5000m-20000m), curved for planet horizon
// drop-off the same way every other curved-world consumer is (terrain.mesh/foliage.mesh/
// particle.mesh/deferred.frag/terrain.glsl's terrainDropOff).
float heightFraction(vec3 worldPos, float distFromCam) {
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;
	return (adjustedY - 5000.0) / 15000.0;
}

// Height gradients for different cloud types: vec4(bottom_fade_start, bottom_fade_end,
// top_fade_start, top_fade_end).
const vec4 stratusGrad       = vec4(0.0, 0.1, 0.2, 0.3);
const vec4 stratocumulusGrad = vec4(0.0, 0.1, 0.5, 0.7);
const vec4 cumulusGrad       = vec4(0.0, 0.1, 0.8, 1.0);

float GetDensityHeightGradientForPoint(float heightFraction, float cloudType) {
	float stratus = max(0.0, 1.0 - cloudType * 2.0);
	float stratocumulus = 1.0 - abs(cloudType - 0.5) * 2.0;
	float cumulus = max(0.0, (cloudType - 0.5) * 2.0);

	vec4 cloudGradient = stratusGrad * stratus +
	                     stratocumulusGrad * stratocumulus +
	                     cumulusGrad * cumulus;

	return smoothstep(cloudGradient.x, cloudGradient.y, heightFraction) -
	       smoothstep(cloudGradient.z, cloudGradient.w, heightFraction);
}

// Samples the shared detail volume (CloudDetailVolume, period CLOUD_VOLUME_PERIOD) at a world
// position, gated to the cloud band on Y via the same heightFraction() every caller uses.
// cloud_shadow_gen.comp used to sample this with a raw worldPos.y/PERIOD wrap instead, putting
// the shadow pass's shapes a different slice of the noise than what's actually rendered.
float sampleCloudVolume(vec3 worldPos, float distFromCam, uint volumeIndex) {
	float uv_y = heightFraction(worldPos, distFromCam);

	// Fixes vertical streaks by preventing Y-axis wrapping.
	if (uv_y < 0.0 || uv_y > 1.0) return 0.0;

	vec3 uvw = fract(worldPos / CLOUD_VOLUME_PERIOD);
	uvw.y = fract(uv_y);

	// Distance-based LOD selection -- alligator()'s sharp Worley-cell edges cause GPU branch
	// divergence without it.
	float volumeLod = clamp(log2(max(1.0, distFromCam / 2000.0)), 0.0, 4.0);
	return SAMPLE_3D_LINEAR_LOD(volumeIndex, uvw, volumeLod).r;
}

// Wind-driven advection speed for cloud sampling (m/s, XZ plane) -- CloudLightingManager's
// cloudFlowSpeed/Direction/HeightScale sliders. `h` is normalized height-in-cloud-band [0,1];
// higher clouds shear ahead of/behind lower ones.
vec3 getCloudAdvectionSpeed(float h) {
	vec2  flowDir = vec2(cos(u_cloudFlowDirection), sin(u_cloudFlowDirection));
	float shear = 1.0 - (h * h) * u_cloudFlowHeightScale;
	return vec3(flowDir.x, 0.0, flowDir.y) * u_cloudFlowSpeed * shear * 10.0;
}

// Shapes a raw 3D-noise sample into final cloud density at a point, given its weather context.
// Shared by cloud_raymarch.comp and cloud_shadow_gen.comp so the shadow pass renders the same
// cloud shapes it's supposed to be casting a shadow for (ported from boidish's
// calculateCloudDensity).
float cloudShapeDensity(float baseNoise, float hFrac, float coverage, float cloudType, float moisture) {
	float shapedCoverage = pow(coverage, remapClamp(hFrac, 0.7, 0.8, 1.0, mix(1.0, 0.5, cloudType)));

	float heightGradient = GetDensityHeightGradientForPoint(hFrac, cloudType);

	// Anvil spreading: strong convective (cumulus-weighted) clouds flatten out near their top
	// instead of tapering to a point -- the classic cumulonimbus silhouette.
	float cumulusWeight = clamp(cloudType * 2.0 - 1.0, 0.0, 1.0);
	float anvilFactor = mix(1.0, 0.25, cumulusWeight * smoothstep(0.6, 0.9, hFrac));
	float coverageThreshold = pow(1.0 - shapedCoverage, remapClamp(hFrac, 0.6, 0.9, 1.0, anvilFactor));

	// Remap-based coverage erosion: treats coverageThreshold as a floor the shaped noise must
	// clear before any density survives, then ramps 0..1 above it.
	float shapedNoise = remapClamp(baseNoise * heightGradient, coverageThreshold, 1.0, 0.0, 1.0);
	return shapedNoise * (0.5 + 1.5 * moisture);
}

// Per-channel extinction coefficient at a point, given its weather context and the artist
// cloudExtinctionColor/Scale sliders (CloudLightingManager). Shared so the shadow pass darkens/
// lightens in step with those sliders instead of using its own independent, hardcoded values.
vec3 cloudExtinctionCoefficient(float hFrac, float cloudType, float moisture) {
	float typeFactor = mix(1.2, 0.7, clamp(cloudType, 0.0, 1.0));
	float bottomMoistureProfile = clamp(1.0 - smoothstep(0.0, 0.85, hFrac), 0.1, 1.0);
	bottomMoistureProfile *= bottomMoistureProfile;
	float baseExtinction = max(
		0.0, (0.005 + 0.02 * moisture) * typeFactor * (0.6 + 0.8 * bottomMoistureProfile)
	);
	return baseExtinction * u_cloudExtinctionColor * u_cloudExtinctionScale;
}

#endif // HELPERS_CLOUD_DENSITY_GLSL
