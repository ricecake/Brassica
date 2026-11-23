#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "common.glsl"
#include "atmosphere/common.glsl"

// Merged cloud + volumetric-fog composite (was CloudCompositeNode + VolumetricRenderNode, two
// separate full-screen passes each blending its own layer directly into HdrColor). Merging them
// is what makes a depth-aware composite ORDER possible at all -- the old two-pass split always
// applied fog last (frontmost) unconditionally, correct from the ground but backwards once fog
// should sit behind a cloud; the old passes never saw each other's un-composited (rgb,
// transmittance) at the same time, so there was nowhere to make that call.
//
// This now splits the fog itself at the cloud's own per-pixel depth (CloudMoments.w) instead of
// picking one global order from camera altitude -- a single altitude threshold can only ever be
// "fog always in front" or "fog always behind" for the WHOLE frame, which pops visibly as the
// camera crosses it and can't represent "this patch of fog is behind that cloud, this other
// patch is in front" at the same time. See sampleFogUpTo/main() below for how.
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CloudCompositePushConstants {
	uint  hdrColorIndex;
	uint  cloudColorIndex;
	uint  cloudTransmittanceIndex;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  cloudMomentsIndex;
	vec4  camForward;
	vec4  camUp;
	vec4  camRight;
	vec4  fovAspect; // x = tanHalfFov, y = aspect
	uvec4 cascadeScatteringIdx;
	uvec4 cascadeExtinctionIdx;
	uint  skyViewIndex;
	float maxDistance;
} push;

float froxelDistToNorm(float dist, float maxDist, float alpha) {
	float normDist = clamp(dist / maxDist, 0.0, 1.0);
	return log(1.0 + normDist * (exp(alpha) - 1.0)) / alpha;
}

vec4 sampleFroxelCascade(int c, vec3 relPos, float dither, out vec3 outTransmittance) {
	float cascadeRanges[3] = float[3](300.0, 3000.0, 100000.0);
	float maxDist = cascadeRanges[c];

	uint scatIdx = push.cascadeScatteringIdx[c];
	uint extIdx = push.cascadeExtinctionIdx[c];

	if (scatIdx == 0xFFFFFFFFu || extIdx == 0xFFFFFFFFu) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	// Project relative position onto camera basis
	float viewZ = dot(relPos, push.camForward.xyz);
	float viewX = dot(relPos, push.camRight.xyz);
	float viewY = dot(relPos, push.camUp.xyz);

	if (viewZ <= 0.001) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	float tanHalfFov = push.fovAspect.x;
	float aspect = push.fovAspect.y;

	float frustumWidthAtZ = 2.0 * viewZ * aspect * tanHalfFov;
	float frustumHeightAtZ = 2.0 * viewZ * tanHalfFov;

	float rayLen = length(relPos);
	float zNorm = froxelDistToNorm(rayLen, maxDist, 3.5);
	// Half-texel offset maps [0..1] range to voxel centers [0.5/64 .. 63.5/64] for smooth trilinear interpolation
	float zTex = (zNorm * 63.0 + 0.5) / 64.0;
	zTex = clamp(zTex + dither, 0.0, 1.0);

	vec3 uv3d = vec3(
		clamp(viewX / frustumWidthAtZ + 0.5, 0.0, 1.0),
		clamp(0.5 - viewY / frustumHeightAtZ, 0.0, 1.0),
		zTex
	);

	vec4 scatSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(scatIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);
	vec4 extSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(extIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);

	outTransmittance = extSample.rgb;
	return scatSample;
}

// Blends the 3 froxel cascades for an arbitrary end-distance along worldRayDir -- same cascade-
// select-and-blend logic a single rayDist sample always used, now reusable so main() can sample
// it twice (once at the real ray end, once at the cloud's own depth) for the depth-split
// composite below. Underwater extinction is scaled by THIS call's own distance `d`, not a
// shared rayDist -- physically correct per segment, and moot either way since cloud_raymarch.comp
// never produces a real cloud while submerged, so there's nothing for the depth split to
// interact with there.
vec3 sampleFogUpTo(vec3 worldRayDir, float d, float ditherOffset, out vec3 outTransmittance) {
	float cascadeRanges[3] = float[3](300.0, 3000.0, 100000.0);
	vec3  relPos = worldRayDir * d;

	vec3 scattering;
	if (d <= cascadeRanges[0]) {
		vec3 t0 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, ditherOffset, t0);
		scattering = s0.rgb;
		outTransmittance = t0;
	} else if (d <= cascadeRanges[1]) {
		float blend = smoothstep(cascadeRanges[0] * 0.7, cascadeRanges[0], d);
		vec3 t0 = vec3(1.0), t1 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, ditherOffset, t0);
		vec4 s1 = sampleFroxelCascade(1, relPos, ditherOffset, t1);
		scattering = mix(s0.rgb, s1.rgb, blend);
		outTransmittance = mix(t0, t1, blend);
	} else {
		float blend = smoothstep(cascadeRanges[1] * 0.7, cascadeRanges[1], d);
		vec3 t1 = vec3(1.0), t2 = vec3(1.0);
		vec4 s1 = sampleFroxelCascade(1, relPos, ditherOffset, t1);
		vec4 s2 = sampleFroxelCascade(2, relPos, ditherOffset, t2);
		scattering = mix(s1.rgb, s2.rgb, blend);
		outTransmittance = mix(t1, t2, blend);
	}

	float camAltKM = uCameraPosition.y / 1000.0;
	float waterLevelKM = u_waterLevel / 1000.0;
	if (camAltKM < waterLevelKM) {
		vec3  waterExtCoeff = vec3(0.45, 0.08, 0.03); // Red attenuated fastest
		float distKM = d / 1000.0;
		vec3  waterTrans = exp(-waterExtCoeff * distKM);
		outTransmittance *= waterTrans;
		scattering *= waterTrans;
	}

	return scattering;
}

void main() {
	vec3 background = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;

	// === Cloud layer (unchanged) ===
	vec3 cloudRgb = SAMPLE_NEAREST(push.cloudColorIndex, inUV).rgb;
	vec3 cloudTransmittance = SAMPLE_NEAREST(push.cloudTransmittanceIndex, inUV).rgb;

	vec4  albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3  gRelPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;
	float cloudDepthMoment = SAMPLE_NEAREST(push.cloudMomentsIndex, inUV).w;
	if (albedo.a >= 0.01 && length(gRelPos) < cloudDepthMoment) {
		cloudTransmittance = vec3(1.0);
		cloudRgb = vec3(0.0);
	}

	// === Fog ray setup (unchanged) ===
	float ditherOffset = (getBayer8x8Dither(ivec2(gl_FragCoord.xy)) - 0.5) / 64.0;
	vec3  relPos = gRelPos;
	float rayDist = length(relPos);
	bool  hasSurface = rayDist > 0.01;

	vec3 worldRayDir;
	if (hasSurface) {
		worldRayDir = relPos / max(1e-3, rayDist);
	} else {
		// Empty sky/clouds/background: reconstruct view ray direction and sample maxDistance
		vec2 clip = inUV * 2.0 - 1.0;
		vec4 viewRay4 = uInvProjMatrix * vec4(clip, 1.0, 1.0);
		vec3 viewDir = viewRay4.xyz / max(1e-4, viewRay4.w);
		worldRayDir = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

		rayDist = push.maxDistance;
		relPos = worldRayDir * rayDist;
	}

	// === Depth-aware fog/cloud split ===
	// The froxel cascades already store FRONT-TO-BACK ACCUMULATED (scattering, transmittance),
	// so sampling the same textures a second time at the cloud's own depth and unwinding the
	// "fresh" remainder segment behind it is enough to interleave correctly -- standard front-
	// to-back accumulation inverse: T(d2) = T(d1) * T_segment, S(d2) = S(d1) + T(d1) * S_segment.
	float camAltKM = uCameraPosition.y / 1000.0;
	float waterLevelKM = u_waterLevel / 1000.0;

	vec3 totalTransmittance;
	vec3 totalScattering = sampleFogUpTo(worldRayDir, rayDist, ditherOffset, totalTransmittance);

	// No real cloud to split around (sentinel depths from cloud_raymarch.comp: 1e9 = ray never
	// hit the cloud shell, -1.0 = a surface blocks the whole shell) -- treat the cloud as being
	// at the far end of the ray, so the "near" segment below becomes the whole fog and the "far"
	// segment degenerates to nothing: today's simple single-layer result.
	float cloudDepth = (cloudDepthMoment > 0.0 && cloudDepthMoment < rayDist) ? cloudDepthMoment : rayDist;

	vec3 nearTransmittance;
	vec3 nearScattering = sampleFogUpTo(worldRayDir, cloudDepth, ditherOffset, nearTransmittance);

	vec3 safeNearTransmittance = max(nearTransmittance, vec3(1e-4));
	// Clamped against sampling noise between the two independent cascade lookups above (T is
	// physically monotonic non-increasing with distance, but these are texture samples, not an
	// exact shared integral) -- never let noise produce transmittance > 1 or negative light.
	vec3 farTransmittance = clamp(totalTransmittance / safeNearTransmittance, vec3(0.0), vec3(1.0));
	vec3 farScattering = max(vec3(0.0), (totalScattering - nearScattering) / safeNearTransmittance);

	// Sky-view LUT: the back-most thing in the stack (true open sky, or whatever's behind a
	// gap) -- folded into the far segment specifically so it naturally ends up behind the cloud
	// in the composite below; the compositing order itself now handles cloud occlusion of it, no
	// manual cloud-transmittance attenuation needed the way the altitude-threshold version did.
	if (!hasSurface && push.skyViewIndex > 0u && push.skyViewIndex != 0xFFFFFFFFu && camAltKM >= waterLevelKM) {
		vec3 skyColor = sampleSkyView(push.skyViewIndex, worldRayDir);
		farScattering += skyColor * farTransmittance * u_skyConvergence;
	}

	// === Composite: far fog (behind cloud) -> cloud -> near fog (in front of cloud) ===
	vec3 result = background * farTransmittance + farScattering;
	result = result * cloudTransmittance + cloudRgb;
	result = result * nearTransmittance + nearScattering;

	outColor = vec4(result, 1.0);
}
