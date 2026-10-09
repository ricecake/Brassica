#ifndef BRASSICA_TERRAIN_DIRTY_GLSL
#define BRASSICA_TERRAIN_DIRTY_GLSL

#include "bindless.glsl"

// Deliberately split out of terrain.glsl: this header is the camera/texel-size math that
// terrain_gen.comp (writes mip 0's dirty strips) and terrain_downsample.comp (rebuilds mips
// 1-10's dirty tiles) both need, and nothing else -- no samplers, no sampleTerrainClipmap/
// ErosionFilter/biome code. Both of those shaders are plain compute, so pulling in terrain.glsl's
// full sampler-heavy surface would needlessly bloat their SPIR-V (and, in a minimal test harness
// that doesn't populate every bindless sampler slot, trip descriptor-indexing validation for
// sampler indices neither shader ever actually touches).

float getLODScale(float lod) {
	// if (lod <= 3.0) {
		return pow(2.0, lod);
	// } else {
	// 	return 8.0 * pow(2.25, lod - 3.0);
	// }
}

float texelSize(
	uint  level
) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	return baseTexelSize * getLODScale(float(level));
}

// -----------------------------------------------------------------------------
// CLIPMAP DIRTY-RECT TRACKING
// -----------------------------------------------------------------------------
//
// Shared between terrain_gen.comp (writes mip 0's dirty strips) and terrain_downsample.comp
// (rebuilds mips 1-10's dirty tiles from it) -- both must derive the exact same storage-space
// dirty region from the same camera delta, or the coarse mip chain silently goes stale relative
// to mip 0.

struct TerrainClipmapUpdate {
	bool  isFull;     // every texel in this layer must be treated as dirty
	ivec2 gridOffset; // camGrid mod dim -- this layer's toroidal wrap origin
	ivec2 delta;      // camGrid - prevCamGrid, in this layer's texels
};

TerrainClipmapUpdate terrainClipmapUpdateFor(uint level, uint textureDim, bool forceRegeneration) {
	float ts = texelSize(level);
	float dim = float((textureDim > 0u) ? textureDim : 1024u);
	int   dimInt = int(dim);

	vec2  camGrid = floor(uCameraPosition.xz / ts);
	vec2  prevCamGrid = floor(uPreviousCameraPosition.xz / ts);
	ivec2 delta = ivec2(camGrid) - ivec2(prevCamGrid);

	TerrainClipmapUpdate u;
	u.gridOffset = ivec2(mod(camGrid, dim));
	u.delta = delta;
	u.isFull =
		forceRegeneration || uFrameIndex == 0u || abs(delta.x) >= dimInt || abs(delta.y) >= dimInt;
	return u;
}

// colIdx/rowIdx are the toroidal-wrapped indices terrain_gen.comp already computes per texel --
// (texelIndex - gridOffset + dim) % dim. Passed in rather than recomputed here so a caller that
// already has them (terrain_gen.comp) doesn't redo the mod, and a caller testing a whole tile
// (terrain_downsample.comp) can pass that tile's corner instead of a single texel.
bool terrainClipmapTexelDirty(TerrainClipmapUpdate u, int colIdx, int rowIdx, uint textureDim) {
	if (u.isFull) {
		return true;
	}
	int dimInt = int((textureDim > 0u) ? textureDim : 1024u);
	int deltaX = u.delta.x;
	int deltaZ = u.delta.y;
	bool dirtyX = (deltaX > 0 && colIdx >= dimInt - deltaX) || (deltaX < 0 && colIdx < -deltaX);
	bool dirtyZ = (deltaZ > 0 && rowIdx >= dimInt - deltaZ) || (deltaZ < 0 && rowIdx < -deltaZ);
	return dirtyX || dirtyZ;
}

#endif // BRASSICA_TERRAIN_DIRTY_GLSL
