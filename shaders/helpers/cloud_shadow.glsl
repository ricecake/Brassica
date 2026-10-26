#ifndef HELPERS_CLOUD_SHADOW_GLSL
#define HELPERS_CLOUD_SHADOW_GLSL

#include "bindless.glsl"
#include "terrain.glsl"

vec4 sampleCloudShadow(uint cloudShadowIndex, uint clipmapIndex, vec3 p, vec3 sunDir, vec3 cameraPos, float shadowMapExtent) {
	if (cloudShadowIndex == 0u || cloudShadowIndex == 0xFFFFFFFFu) return vec4(0.0, 0.0, 0.0, 1.0);
	float sunLen = length(sunDir);
	if (sunLen < 0.001) return vec4(0.0, 0.0, 0.0, 1.0);
	vec3 sDir = sunDir / sunLen;
	if (sDir.y <= 0.001) return vec4(0.0, 0.0, 100.0, 0.0);

	float raySunY = max(sDir.y, 0.05);

	float dropOff = terrainDropOff(p.xz);
	float terrainH = (clipmapIndex > 0u && clipmapIndex != 0xFFFFFFFFu) ? sampleTerrainClipmap(clipmapIndex, p.xz, 0u, 1088u).r : 0.0;
	float groundY = terrainH - dropOff;

	vec2 projXZ = p.xz - (max(0.0, p.y - groundY) / raySunY) * sDir.xz;
	vec2 centerXZ = cameraPos.xz - sDir.xz * clamp(10000.0 / raySunY, 0.0, 100000.0);

	vec2 shadowUV = (projXZ - centerXZ) / shadowMapExtent + 0.5;
	shadowUV = clamp(shadowUV, 0.0005, 0.9995);

	vec4 s0_top = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 0.0));
	float cloudStartHeight = s0_top.r;
	float cloudEndHeight = s0_top.g;

	if (p.y >= cloudStartHeight) {
		return vec4(0.0, 0.0, 0.0, 1.0);
	}

	if (p.y <= cloudEndHeight) {
		return SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 4.0));
	}

	float activeRange = max(1.0, cloudStartHeight - cloudEndHeight);
	float hFraction = clamp((cloudStartHeight - p.y) / activeRange, 0.0, 1.0);

	vec4 s_entrance = vec4(0.0, 0.0, 0.0, 1.0);

	vec4 sa, sb;
	float localT = 0.0;

	if (hFraction <= 0.25) {
		sa = s_entrance;
		sb = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 1.0));
		localT = hFraction / 0.25;
	} else if (hFraction <= 0.50) {
		sa = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 1.0));
		sb = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 2.0));
		localT = (hFraction - 0.25) / 0.25;
	} else if (hFraction <= 0.75) {
		sa = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 2.0));
		sb = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 3.0));
		localT = (hFraction - 0.50) / 0.25;
	} else {
		sa = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 3.0));
		sb = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 4.0));
		localT = (hFraction - 0.75) / 0.25;
	}

	return mix(sa, sb, clamp(localT, 0.0, 1.0));
}

vec4 sampleCloudShadow(uint cloudShadowIndex, vec3 p, vec3 sunDir, vec3 cameraPos, float shadowMapExtent) {
	return sampleCloudShadow(cloudShadowIndex, 0u, p, sunDir, cameraPos, shadowMapExtent);
}

#endif // HELPERS_CLOUD_SHADOW_GLSL
