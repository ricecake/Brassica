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

	float y = p.y;
	vec4 s0, s1;
	float mixVal = 0.0;
	if (y >= 20000.0) {
		return SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 0.0));
	} else if (y >= 12500.0) {
		s0 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 0.0));
		s1 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 1.0));
		mixVal = (20000.0 - y) / 7500.0;
	} else if (y >= 5000.0) {
		s0 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 1.0));
		s1 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 2.0));
		mixVal = (12500.0 - y) / 7500.0;
	} else if (y >= 2500.0) {
		s0 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 2.0));
		s1 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 3.0));
		mixVal = (5000.0 - y) / 2500.0;
	} else {
		s0 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 3.0));
		s1 = SAMPLE_ARRAY_CLAMP(cloudShadowIndex, vec3(shadowUV, 4.0));
		mixVal = (2500.0 - clamp(y, groundY, 2500.0)) / max(1.0, 2500.0 - groundY);
	}
	return mix(s0, s1, clamp(mixVal, 0.0, 1.0));
}

vec4 sampleCloudShadow(uint cloudShadowIndex, vec3 p, vec3 sunDir, vec3 cameraPos, float shadowMapExtent) {
	return sampleCloudShadow(cloudShadowIndex, 0u, p, sunDir, cameraPos, shadowMapExtent);
}

#endif // HELPERS_CLOUD_SHADOW_GLSL
