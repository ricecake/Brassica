#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"

float getLODScale(float lod) {
	if (lod <= 3.0) {
		return pow(2.0, lod);
	} else {
		return 8.0 * pow(2.25, lod - 3.0);
	}
}

// Toroidal UV mapping helper for terrain clipmap textures
vec2 sampleToroidalUV(
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float texelSize = baseTexelSize * getLODScale(float(level));
	uint  dim = (textureDim > 0u) ? textureDim : 1088u;

	uvec4 offsets = (level < 4u) ? lodOffsets0_3 : ((level < 8u) ? lodOffsets4_7 : lodOffsets8_11);
	uint  packed = offsets[level % 4u];
	ivec2 gridOffset = ivec2(int(packed & 0xFFFFu), int((packed >> 16u) & 0xFFFFu));

	vec2 centerWorldPos = floor(uCameraPosition.xz / texelSize) * texelSize;
	vec2 deltaWorld = worldXZ - centerWorldPos;
	vec2 texelCoord = deltaWorld / texelSize + vec2(float(dim) * 0.5) + vec2(gridOffset);

	return fract(texelCoord / float(dim));
}

vec2 sampleToroidalUV(vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	return sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

// Sample terrain height (r) and normal (gba) from clipmap
vec4 sampleTerrainClipmap(
	uint  clipmapIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11);
	return SAMPLE_ARRAY_WRAP(clipmapIndex, vec3(uv, float(level)));
}

vec4 sampleTerrainClipmap(
	uint  clipmapIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainClipmap(clipmapIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

// Sample terrain min-max height map (r = min height, g = max height)
vec2 sampleTerrainMinMax(
	uint  minMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11);
	return SAMPLE_ARRAY_WRAP(minMaxIndex, vec3(uv, float(level))).rg;
}

vec2 sampleTerrainMinMax(
	uint  minMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainMinMax(minMaxIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

// Sample terrain biome map (r = biome weight/type, g = terrain variance for LOD adjustments, b = detail, a = moisture)
vec4 sampleTerrainBiome(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11);
	return SAMPLE_ARRAY_WRAP(biomeIndex, vec3(uv, float(level)));
}

vec4 sampleTerrainBiome(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainBiome(biomeIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

// Sample terrain tile visibility map (r = shore distance, g = flow strength, b = visibility flag, a = land mask)
vec4 sampleTerrainTileVisibility(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11);
	return SAMPLE_ARRAY_WRAP(visIndex, vec3(uv, float(level)));
}

vec4 sampleTerrainTileVisibility(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainTileVisibility(visIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

// Helper functions for sampling fluid flow map and shore distance
vec2 sampleTerrainFlow(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	return sampleTerrainBiome(biomeIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11).ba;
}

vec2 sampleTerrainFlow(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainFlow(biomeIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

float sampleTerrainShoreDistance(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	return sampleTerrainTileVisibility(visIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11).r;
}

float sampleTerrainShoreDistance(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainShoreDistance(visIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
}

#endif // BRASSICA_TERRAIN_GLSL
