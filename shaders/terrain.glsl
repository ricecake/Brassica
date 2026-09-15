#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"

struct TerrainTileSample {
	uint tileIndex;
	vec2 localUV;
};

// Quadtree Indirection Map lookup for terrain tile sampling
TerrainTileSample getTerrainTileSampleUV(uint indirectionIndex, vec2 worldXZ, uint level) {
	vec2 rootMin = vec2(-32768.0);
	float rootExtent = 65536.0;
	vec2 indirectionUV = (worldXZ - rootMin) / rootExtent;

	TerrainTileSample res;
	res.tileIndex = 65535u;
	res.localUV = vec2(0.0);

	float baseTileExtent = 512.0;
	uint startLOD = min(level, 7u);

	for (uint l = startLOD; l < 8u; ++l) {
		float gridSize = float(128u >> l);
		ivec2 gridCoord = clamp(ivec2(floor(indirectionUV * gridSize)), ivec2(0), ivec2(int(gridSize) - 1));
		vec2 mipUV = (vec2(gridCoord) + vec2(0.5)) / gridSize;
		float val = textureLod(sampler2D(uTextures2D[nonuniformEXT(indirectionIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), mipUV, float(l)).r;
		uint idx = uint(round(val));
		if (idx < 65535u) {
			res.tileIndex = idx;
			float tileSize = baseTileExtent * pow(2.0, float(l));
			vec2 tileMin = rootMin + vec2(gridCoord) * tileSize;
			res.localUV = clamp((worldXZ - tileMin) / tileSize, vec2(0.0), vec2(1.0));
			break;
		}
	}
	return res;
}

// Sample terrain height (r) and normal (gba) using indirection map
vec4 sampleTerrainClipmap(uint clipmapIndex, uint indirectionIndex, vec2 worldXZ, uint level) {
	TerrainTileSample sampleInfo = getTerrainTileSampleUV(indirectionIndex, worldXZ, level);
	if (sampleInfo.tileIndex >= 65535u) return vec4(0.0);
	return SAMPLE_ARRAY_WRAP(clipmapIndex, vec3(sampleInfo.localUV, float(sampleInfo.tileIndex)));
}

// Sample terrain min-max height map (r = min height, g = max height)
vec2 sampleTerrainMinMax(uint minMaxIndex, uint indirectionIndex, vec2 worldXZ, uint level) {
	TerrainTileSample sampleInfo = getTerrainTileSampleUV(indirectionIndex, worldXZ, level);
	if (sampleInfo.tileIndex >= 65535u) return vec2(0.0);
	return SAMPLE_ARRAY_WRAP(minMaxIndex, vec3(sampleInfo.localUV, float(sampleInfo.tileIndex))).rg;
}

// Sample terrain biome map
vec4 sampleTerrainBiome(uint biomeIndex, uint indirectionIndex, vec2 worldXZ, uint level) {
	TerrainTileSample sampleInfo = getTerrainTileSampleUV(indirectionIndex, worldXZ, level);
	if (sampleInfo.tileIndex >= 65535u) return vec4(0.0);
	return SAMPLE_ARRAY_WRAP(biomeIndex, vec3(sampleInfo.localUV, float(sampleInfo.tileIndex)));
}

// Sample terrain tile visibility map
float sampleTerrainTileVisibility(uint visIndex, uint indirectionIndex, vec2 worldXZ, uint level) {
	TerrainTileSample sampleInfo = getTerrainTileSampleUV(indirectionIndex, worldXZ, level);
	if (sampleInfo.tileIndex >= 65535u) return 1.0;
	return SAMPLE_ARRAY_WRAP(visIndex, vec3(sampleInfo.localUV, float(sampleInfo.tileIndex))).r;
}

// Legacy signature overloads for backward compatibility
vec4 sampleTerrainClipmap(uint clipmapIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	return sampleTerrainClipmap(clipmapIndex, 0u, worldXZ, level);
}

vec2 sampleTerrainMinMax(uint minMaxIndex, vec2 worldXZ, uint level, uint textureDim, uvec4 lodOffsets0_3, uvec4 lodOffsets4_7) {
	return sampleTerrainMinMax(minMaxIndex, 0u, worldXZ, level);
}

#endif // BRASSICA_TERRAIN_GLSL
