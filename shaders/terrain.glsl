#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"

// Toroidal UV mapping helper for terrain clipmap textures
vec2 sampleToroidalUV(vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float texelSize = baseTexelSize * pow(2.0, float(level));
	uint dim = (textureDim > 0u) ? textureDim : 1088u;

	uvec4 offsets = (level < 4u) ? lodOffsets0_3 : lodOffsets4_7;
	uint packed = offsets[level % 4u];
	ivec2 gridOffset = ivec2(int(packed & 0xFFFFu), int((packed >> 16u) & 0xFFFFu));

	vec2 centerWorldPos = floor(uCameraPosition.xz / texelSize) * texelSize;
	vec2 deltaWorld = worldXZ - centerWorldPos;
	vec2 texelCoord = deltaWorld / texelSize + vec2(float(dim) * 0.5) + vec2(gridOffset);

	return fract(texelCoord / float(dim));
}

// Sample terrain height (r) and normal (gba) from clipmap
vec4 sampleTerrainClipmap(uint clipmapIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7);
	return SAMPLE_ARRAY_WRAP(clipmapIndex, vec3(uv, float(level)));
}

// Sample terrain min-max height map (r = min height, g = max height)
vec2 sampleTerrainMinMax(uint minMaxIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7);
	return SAMPLE_ARRAY_WRAP(minMaxIndex, vec3(uv, float(level))).rg;
}

// Sample terrain biome map (r = biome weight/type, g = terrain variance for LOD adjustments, b = detail, a = moisture)
vec4 sampleTerrainBiome(uint biomeIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7);
	return SAMPLE_ARRAY_WRAP(biomeIndex, vec3(uv, float(level)));
}

// Sample terrain tile visibility map (r = visibility flag / occlusion factor)
float sampleTerrainTileVisibility(uint visIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7);
	return SAMPLE_ARRAY_WRAP(visIndex, vec3(uv, float(level))).r;
}

#endif // BRASSICA_TERRAIN_GLSL
