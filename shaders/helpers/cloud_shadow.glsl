#ifndef HELPERS_CLOUD_SHADOW_GLSL
#define HELPERS_CLOUD_SHADOW_GLSL

#include "bindless.glsl"
#include "helpers/cloud_shadow_mapping.glsl"

// No clipmapIndex parameter anymore -- the old terrain-height-based projection needed a terrain
// sample to find where to project p down to, and that lookup only ever used clipmap level 0,
// valid within ~245m of the camera against this map's 320km extent (see
// helpers/cloud_shadow_mapping.glsl's comment on cloudShadowLabel). The flat-plane label needs
// no terrain sample at all, and -- unlike the old approach -- is guaranteed to agree with what
// the writer (cloud_shadow_gen.comp) used for this exact point, since both call the same
// function.
vec4 sampleCloudShadow(uint cloudShadowIndex, vec3 p, vec3 sunDir, vec3 cameraPos, float shadowMapExtent) {
	if (cloudShadowIndex == 0u || cloudShadowIndex == 0xFFFFFFFFu) return vec4(0.0, 0.0, 0.0, 1.0);
	float sunLen = length(sunDir);
	if (sunLen < 0.001) return vec4(0.0, 0.0, 0.0, 1.0);
	vec3 sDir = sunDir / sunLen;
	if (sDir.y <= 0.001) return vec4(0.0, 0.0, 100.0, 0.0);

	float raySunY = max(sDir.y, 0.05);

	vec2 label = cloudShadowLabel(p, sDir, raySunY);
	vec2 shadowUV = cloudShadowUV(label, shadowMapExtent);

	vec4 s0_top = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 0.0));
	float cloudStartHeight = s0_top.r;
	float cloudEndHeight = s0_top.g;

	if (p.y >= cloudStartHeight) {
		return vec4(0.0, 0.0, 0.0, 1.0);
	}

	if (p.y <= cloudEndHeight) {
		return SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 4.0));
	}

	float activeRange = max(1.0, cloudStartHeight - cloudEndHeight);
	float hFraction = clamp((cloudStartHeight - p.y) / activeRange, 0.0, 1.0);

	vec4 s_entrance = vec4(0.0, 0.0, 0.0, 1.0);

	vec4 sa, sb;
	float localT = 0.0;

	if (hFraction <= 0.25) {
		sa = s_entrance;
		sb = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 1.0));
		localT = hFraction / 0.25;
	} else if (hFraction <= 0.50) {
		sa = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 1.0));
		sb = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 2.0));
		localT = (hFraction - 0.25) / 0.25;
	} else if (hFraction <= 0.75) {
		sa = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 2.0));
		sb = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 3.0));
		localT = (hFraction - 0.50) / 0.25;
	} else {
		sa = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 3.0));
		sb = SAMPLE_ARRAY_WRAP(cloudShadowIndex, vec3(shadowUV, 4.0));
		localT = (hFraction - 0.75) / 0.25;
	}

	return mix(sa, sb, clamp(localT, 0.0, 1.0));
}

#endif // HELPERS_CLOUD_SHADOW_GLSL
