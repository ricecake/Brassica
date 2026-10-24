#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"
#include "helpers/culling.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/whittaker.glsl"
#include "terrain_dirty.glsl"

// Converts 2D UV within a cubemap face [0, 1]^2 and face index [0..5] to 3D unit direction vector
vec3 cubeFaceUVToDirection(vec2 uv, uint face) {
	vec2 st = uv * 2.0 - 1.0;
	vec3 dir;
	if (face == 0u)      dir = vec3(1.0, -st.y, -st.x);
	else if (face == 1u) dir = vec3(-1.0, -st.y, st.x);
	else if (face == 2u) dir = vec3(st.x, 1.0, st.y);
	else if (face == 3u) dir = vec3(st.x, -1.0, -st.y);
	else if (face == 4u) dir = vec3(st.x, -st.y, 1.0);
	else                 dir = vec3(-st.x, -st.y, -1.0);
	return normalize(dir);
}

// Converts 3D unit direction vector to 2D UV within cubemap face [0, 1]^2 and face index [0..5] stored in z channel
vec3 directionToCubeFaceUV(vec3 dir) {
	vec3 absDir = abs(dir);
	float ma;
	vec2 uv;
	float face;
	if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
		ma = absDir.x;
		if (dir.x > 0.0) {
			face = 0.0;
			uv = vec2(-dir.z, -dir.y);
		} else {
			face = 1.0;
			uv = vec2(dir.z, -dir.y);
		}
	} else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
		ma = absDir.y;
		if (dir.y > 0.0) {
			face = 2.0;
			uv = vec2(dir.x, dir.z);
		} else {
			face = 3.0;
			uv = vec2(dir.x, -dir.z);
		}
	} else {
		ma = absDir.z;
		if (dir.z > 0.0) {
			face = 4.0;
			uv = vec2(dir.x, -dir.y);
		} else {
			face = 5.0;
			uv = vec2(-dir.x, -dir.y);
		}
	}
	uv = (uv / ma + 1.0) * 0.5;
	return vec3(uv, face);
}

// Toroidal UV mapping helper for terrain clipmap textures
vec2 sampleToroidalUV(vec2 worldXZ, uint level, uint textureDim) {
    float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
    float texelSize = baseTexelSize * getLODScale(float(level));
    float dim = float((textureDim > 0u) ? textureDim : 1088u);

    // Remove floor() to preserve continuous sub-texel precision
    vec2 worldGrid = worldXZ / texelSize;

    vec2 texelCoord = mod(worldGrid + dim * 0.5, dim);

    // Add 0.5 to shift from texel corners to texel centers
    return (texelCoord + 0.5) / dim;
}

// Sample terrain clipmap (r = height, g = variance, ba = octahedral encoded normal)
vec4 sampleTerrainClipmap(
	uint  clipmapIndex,
	uint  cubemapIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	if (level >= 8u && cubemapIndex > 0u && cubemapIndex != 0xFFFFFFFFu) {
		float theta = worldXZ.x / FAKE_PLANET_RADIUS;
		float phi   = worldXZ.y / FAKE_PLANET_RADIUS;
		vec3 dir    = vec3(cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta));
		vec3 faceUV = directionToCubeFaceUV(dir);
		return SAMPLE_ARRAY_WRAP(cubemapIndex, vec3(faceUV.xy, faceUV.z));
	}
	uint clampedLevel = min(level, 7u);
	vec2 uv = sampleToroidalUV(worldXZ, clampedLevel, textureDim);
	return SAMPLE_ARRAY_WRAP(clipmapIndex, vec3(uv, float(clampedLevel)));
}

vec4 sampleTerrainClipmap(
	uint  clipmapIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	return sampleTerrainClipmap(clipmapIndex, 0u, worldXZ, level, textureDim);
}

// Sample terrain min-max height map (r = min height, g = max height)
vec2 sampleTerrainMinMax(
	uint  minMaxIndex,
	uint  cubemapMinMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	if (level >= 8u && cubemapMinMaxIndex > 0u && cubemapMinMaxIndex != 0xFFFFFFFFu) {
		float theta = worldXZ.x / FAKE_PLANET_RADIUS;
		float phi   = worldXZ.y / FAKE_PLANET_RADIUS;
		vec3 dir    = vec3(cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta));
		vec3 faceUV = directionToCubeFaceUV(dir);
		return SAMPLE_ARRAY_WRAP(cubemapMinMaxIndex, vec3(faceUV.xy, faceUV.z)).rg;
	}
	uint clampedLevel = min(level, 7u);
	vec2 uv = sampleToroidalUV(worldXZ, clampedLevel, textureDim);
	return SAMPLE_ARRAY_WRAP(minMaxIndex, vec3(uv, float(clampedLevel))).rg;
}

vec2 sampleTerrainMinMax(
	uint  minMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	return sampleTerrainMinMax(minMaxIndex, 0u, worldXZ, level, textureDim);
}

// Sample terrain biome map (r = biome, g = water level, b = erosion ridge map, a = substrate map)
vec4 sampleTerrainBiome(
	uint  biomeIndex,
	uint  cubemapBiomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	if (level >= 8u && cubemapBiomeIndex > 0u && cubemapBiomeIndex != 0xFFFFFFFFu) {
		float theta = worldXZ.x / FAKE_PLANET_RADIUS;
		float phi   = worldXZ.y / FAKE_PLANET_RADIUS;
		vec3 dir    = vec3(cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta));
		vec3 faceUV = directionToCubeFaceUV(dir);
		return SAMPLE_ARRAY_WRAP(cubemapBiomeIndex, vec3(faceUV.xy, faceUV.z));
	}
	uint clampedLevel = min(level, 7u);
	vec2 uv = sampleToroidalUV(worldXZ, clampedLevel, textureDim);
	return SAMPLE_ARRAY_WRAP(biomeIndex, vec3(uv, float(clampedLevel)));
}

vec4 sampleTerrainBiome(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	return sampleTerrainBiome(biomeIndex, 0u, worldXZ, level, textureDim);
}

// Sample planar weather and Whittaker biome texture
vec4 sampleTerrainWeatherBiome(uint weatherStorageIdx, vec3 worldPos) {
	float worldExtent = 2.0 * FAKE_PLANET_RADIUS * 3.14159265359;
	vec2 uv = (worldPos.xz / worldExtent) + 0.5;
	return SAMPLE_LINEAR(weatherStorageIdx, fract(uv));
}

// -----------------------------------------------------------------------------
// TERRAIN OCCLUSION
// -----------------------------------------------------------------------------
//
// A real hierarchical min-max traversal against the clipmap's mip chain -- a quadtree-style
// march that climbs to coarser mips (and, past a layer's own safety window, coarser clipmap
// layers) while cells read as fully clear, and descends again wherever a cell's bound is
// ambiguous, instead of a flat fixed-step march. One shared core (traceTerrainOcclusion) plus
// two thin wrappers matching the two shapes callers need: an AABB test and a direct
// origin/direction/distance test. Horizon-drop aware exactly: every height comparison is made
// against the ray's exact convex curvature profile over the step in question (terrainRaySegmentMinG
// below), not a per-sample approximation -- the same worldY -= d^2/(2*FAKE_PLANET_RADIUS)
// correction every other curved-world consumer uses (terrain.mesh/foliage.mesh/particle.mesh/
// deferred.frag), just minimized exactly over each step instead of sampled at its endpoints.
// isBeyondHorizon (helpers/culling.glsl) is still used as a cheap full-reject in the AABB
// wrapper before any traversal happens.

#ifndef TERRAIN_OCCLUSION_MAX_ITERATIONS
#define TERRAIN_OCCLUSION_MAX_ITERATIONS 64
#endif

// Bundles the indices/counts every occlusion call needs so call sites pass one value instead
// of several. refHeight is the reference surface isBeyondHorizon measures altitude from --
// pass the water level (or 0 on dry maps). No clipmapIndex: the hierarchical tracer below never
// needs an exact clipmap sample, only the min/max chain, so there's nothing for it to read.
struct TerrainOcclusionContext {
	uint  minMaxIndex;
	uint  cubemapMinMaxIndex;
	uint  textureDim;
	uint  numLevels;
	float refHeight;
};

TerrainOcclusionContext makeTerrainOcclusionContext(
	uint  minMaxIndex,
	uint  cubemapMinMaxIndex,
	uint  textureDim,
	uint  numLevels,
	float refHeight
) {
	TerrainOcclusionContext ctx;
	ctx.minMaxIndex = minMaxIndex;
	ctx.cubemapMinMaxIndex = cubemapMinMaxIndex;
	ctx.textureDim = textureDim;
	ctx.numLevels = numLevels;
	ctx.refHeight = refHeight;
	return ctx;
}

TerrainOcclusionContext makeTerrainOcclusionContext(
	uint  minMaxIndex,
	uint  textureDim,
	uint  numLevels,
	float refHeight
) {
	return makeTerrainOcclusionContext(minMaxIndex, 0u, textureDim, numLevels, refHeight);
}

// Explicit-mip counterpart to sampleTerrainMinMax above. SAMPLE_ARRAY_WRAP's texture()
// call has no LOD argument, so in any stage without automatic fragment derivatives
// (compute/task/mesh -- every current and planned caller of this file's occlusion helpers)
// it always resolves to mip 0, silently skipping every coarser level
// TerrainMinMaxDownsampleNode built. texelFetch (not a filtered sample -- interpolating
// two "max" texels would no longer be a valid conservative bound) against an explicit mip
// is what actually reaches those levels.
vec2 sampleTerrainMinMaxMip(
	uint  minMaxIndex,
	uint  cubemapMinMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  mip,
	uint  textureDim
) {
	if (level >= 8u && cubemapMinMaxIndex > 0u && cubemapMinMaxIndex != 0xFFFFFFFFu) {
		float theta = worldXZ.x / FAKE_PLANET_RADIUS;
		float phi   = worldXZ.y / FAKE_PLANET_RADIUS;
		vec3 dir    = vec3(cos(phi) * sin(theta), sin(phi), cos(phi) * cos(theta));
		vec3 faceUV = directionToCubeFaceUV(dir);

		int mipDim = max(1, 8192 >> int(mip));
		ivec2 mipTexel = clamp(ivec2(faceUV.xy * float(mipDim)), ivec2(0), ivec2(mipDim - 1));

		return texelFetch(
			sampler2DArray(uTextureArrays[nonuniformEXT(cubemapMinMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			ivec3(mipTexel, int(faceUV.z)),
			int(mip)
		).rg;
	}

	uint clampedLevel = min(level, 7u);
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float ts = baseTexelSize * getLODScale(float(clampedLevel));

	if (mip == 0u) {
		float dim = float((textureDim > 0u) ? textureDim : 1088u);
		vec2 worldGrid = floor(worldXZ / ts);
		vec2 texelCoord = mod(worldGrid + dim * 0.5, dim);

		ivec2 pix = ivec2(texelCoord);
		ivec2 pixNext = ivec2(mod(texelCoord + 1.0, dim));
		vec2 a = texelFetch(
			sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			ivec3(pix.x, pix.y, int(clampedLevel)), 0
		).rg;
		vec2 b = texelFetch(
			sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			ivec3(pixNext.x, pix.y, int(clampedLevel)), 0
		).rg;
		vec2 c = texelFetch(
			sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			ivec3(pix.x, pixNext.y, int(clampedLevel)), 0
		).rg;
		vec2 d = texelFetch(
			sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			ivec3(pixNext.x, pixNext.y, int(clampedLevel)), 0
		).rg;
		return vec2(min(min(a.x, b.x), min(c.x, d.x)), max(max(a.y, b.y), max(c.y, d.y)));
	}

	float downsampleDim = 1024.0;
	vec2 worldGrid = floor(worldXZ / ts);
	vec2 texelCoord = mod(worldGrid + downsampleDim * 0.5, downsampleDim);

	int  mipDim = max(1, int(downsampleDim) >> int(mip));
	ivec2 mipTexel = clamp(ivec2(texelCoord) >> int(mip), ivec2(0), ivec2(mipDim - 1));

	return texelFetch(
		sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		ivec3(mipTexel, int(clampedLevel)),
		int(mip)
	).rg;
}

vec2 sampleTerrainMinMaxMip(
	uint  minMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  mip,
	uint  textureDim
) {
	return sampleTerrainMinMaxMip(minMaxIndex, 0u, worldXZ, level, mip, textureDim);
}

// Half-extent (world units) of level L's 0.9-safety Chebyshev window around the camera --
// shared with the hierarchical tracer below, which needs every level's window, not just
// level 0's.
float terrainLevelHalfExtent(uint level, uint textureDim) {
	const float kClipmapCoverageSafety = 0.9; // leaves slack for snapping/update lag
	float dim = float((textureDim > 0u) ? textureDim : 1088u);
	return 0.5 * dim * texelSize(level) * kClipmapCoverageSafety;
}

// Camera-centered LOD pick for a world-space point, generalizing deferred.frag's
// calculateRayLOD (which hardcodes baseRadius=272.0 for its own short reflection-probe
// range) into something usable for an arbitrarily long occlusion march. Returns
// numLevels itself as an "out of clipmap coverage" sentinel for the march to stop on.
uint terrainClipmapLevelFor(vec2 worldXZ, uint textureDim, uint numLevels) {
	float halfExtent0 = terrainLevelHalfExtent(0u, textureDim);

	vec2  d2 = abs(worldXZ - uCameraPosition.xz);
	float d = max(d2.x, d2.y);
	if (d <= halfExtent0) {
		return 0u;
	}
	uint level = uint(max(ceil(log2(d / halfExtent0)), 1.0));
	return min(level, numLevels);
}

// Same curvature correction every other curved-world consumer applies at its own sample
// points (terrain.mesh/foliage.mesh/particle.mesh/deferred.frag) -- kept camera-centered
// here too, consistent with how the world is actually drawn.
float terrainDropOff(vec2 worldXZ) {
	vec2 flatXZ = worldXZ - uCameraPosition.xz;
	return dot(flatXZ, flatXZ) / (2.0 * FAKE_PLANET_RADIUS);
}

// Absolute ray parameter t (origin + dir*t) at which the ray exits the single axis-aligned
// cell of width cellSize that curPos currently sits in, along one axis. 1e30 ("never", along
// this axis) if dir has ~no component on that axis -- the other axis's exit still bounds t.
float terrainAxisExitT(float originAxis, float dirAxis, float curPos, float cellSize) {
	if (abs(dirAxis) < 1e-9) {
		return 1e30;
	}
	float cellIdx = floor(curPos / cellSize);
	float boundary = (dirAxis > 0.0) ? (cellIdx + 1.0) * cellSize : cellIdx * cellSize;
	return (boundary - originAxis) / dirAxis;
}

// 2D (XZ) cell exit, both axes combined -- the standard DDA "which wall do we cross first" step.
float terrainCellExitT(vec3 origin, vec3 dir, vec2 curXZ, float cellSize) {
	return min(
		terrainAxisExitT(origin.x, dir.x, curXZ.x, cellSize),
		terrainAxisExitT(origin.z, dir.z, curXZ.y, cellSize)
	);
}

// Same idea for a whole layer's safety window (a square centered on the camera, not a single
// traversal cell) -- recomputed once per layer the ray is in, not once per cell.
float terrainLayerExitT(vec3 origin, vec3 dir, vec2 center, float halfExtent) {
	float tx = (abs(dir.x) < 1e-9)
		? 1e30
		: (((dir.x > 0.0 ? center.x + halfExtent : center.x - halfExtent) - origin.x) / dir.x);
	float tz = (abs(dir.z) < 1e-9)
		? 1e30
		: (((dir.z > 0.0 ? center.y + halfExtent : center.y - halfExtent) - origin.z) / dir.z);
	return min(tx, tz);
}

// g(t) = C + B*t + A*t^2 is the ray's render-space height (including the curved-world drop)
// at parameter t, exactly -- see terrainMarchStep for the A/B/C derivation. Convex since
// A = |dir.xz|^2 / (2*FAKE_PLANET_RADIUS) >= 0, so its minimum over [t0, t1] is at the vertex
// clamped into the interval (or, in the degenerate near-vertical/A~0 case, at whichever endpoint
// is lower -- g is then effectively linear in t).
float terrainRaySegmentMinG(float A, float B, float C, float t0, float t1) {
	float tStar = (A > 1e-12) ? clamp(-B / (2.0 * A), t0, t1) : (B >= 0.0 ? t0 : t1);
	return C + B * tStar + A * tStar * tStar;
}

// Shared march core for both consumers below (traceTerrainOcclusion's hard occlusion test and
// calculateTerrainSoftShadow's penumbra), so the cell/mip/layer stepping lives in exactly one
// place instead of two independently-hand-rolled copies.
//
// m is indexed against SPD's own dilated mips, offset by one (m=0 reads TerrainMinMaxTexture
// mip 1, m=-1 reads mip 0) -- cellSize = texelSize(L) * 2^(m+1) and
// sampleTerrainMinMaxMip(..., m+1) both fall out of the same formula at m=-1 (2^0 = 1 texel,
// mip 0) as at m>=0, no special-casing needed. mip 0 is TerrainGenNode's raw per-texel height,
// with no dilation baked in on write -- unlike every other mip, which SpdLoadSourceImage builds
// pre-dilated. sampleTerrainMinMaxMip dilates mip 0 on the read side instead (its own comment),
// so cellSize and the queried footprint still match at every step: a 1-texel-wide DDA step at
// m=-1 is bounded by exactly the two corners (pix, pix+1) that step crosses between, the same
// shape of match mip 1's own build already guarantees for m=0. An earlier version of this file
// used raw, undilated mip 0 as its finest level directly, which was a real mismatch (the DDA
// stepped it as 1 texel wide while the ad-hoc dilation attempt covered more than that) and
// showed up as inconsistent per-texel-corner results -- not the same thing as what m=-1 does
// here, which keeps the one-step/one-dilation-span invariant every other mip already relies on.
//
// Resolves "ambiguous" (cell bound straddles the ray) by descending internally in its own loop,
// so callers never see that state -- they only ever get HIT or ADVANCE (CLEAR, or ambiguous even
// at mip 0, treated identically: never register a grazing ridge as a hit, advance instead). On
// ADVANCE, t0/t1 bound the segment just evaluated and clearance is a sentinel (-1.0, "no
// reliable margin") unless the segment was genuinely, confidently clear (gMin above the cell's
// max by more than kEps), in which case it's gMin's real positive margin above that max --
// callers must check clearance >= 0.0 before trusting it for anything, rather than treating the
// old 0-via-clamp value as if it meant "zero margin but still clear" (it didn't: it covered
// ambiguous and layer-transition-bypassed segments too, which calculateTerrainSoftShadow used to
// silently treat as full occlusion before this was a sentinel -- see its own comment).
#define TERRAIN_MARCH_HIT 0
#define TERRAIN_MARCH_ADVANCE 1

int terrainMarchStep(
	TerrainOcclusionContext ctx,
	vec3  origin,
	vec3  d,
	float A,
	float B,
	float C,
	vec2  camXZ,
	float maxDistance,
	inout uint  L,
	inout int   m,
	inout float t,
	out float t0,
	out float t1,
	out float clearance
) {
	const float kEps = 0.1;    // bias toward "ambiguous" (resolved by descending) over a wrong call
	const float kNudge = 1e-3; // push just past a cell/layer boundary so the next floor() is unambiguous

	t0 = t;
	clearance = -1.0; // sentinel: "no reliable margin" until a genuinely clear branch sets it

	// Bounded by the mip range itself (m descends at most 11 times, mip 0..10) -- not a
	// heuristic cap, just loud insurance against ever spinning here forever.
	for (int descend = 0; descend < 12; ++descend) {
		float cellSize = texelSize(L) * exp2(float(m + 1));
		vec3  p = origin + d * t;

		float cellExitT = terrainCellExitT(origin, d, p.xz, cellSize);
		float layerHalfExtent = terrainLevelHalfExtent(L, ctx.textureDim);
		float layerExitT = terrainLayerExitT(origin, d, camXZ, layerHalfExtent);
		bool  exitingLayer = layerExitT <= cellExitT;
		float tEnd = min(min(cellExitT, layerExitT), maxDistance);
		if (tEnd <= t) {
			tEnd = t + cellSize * 1e-4; // degenerate (e.g. dir parallel to a cell edge through its corner)
		}
		t1 = tEnd;

		vec2  cellMinMax = sampleTerrainMinMaxMip(ctx.minMaxIndex, ctx.cubemapMinMaxIndex, p.xz, L, uint(m + 1), ctx.textureDim);
		float gMin = terrainRaySegmentMinG(A, B, C, t, tEnd);

		// Exiting the layer's own safety window takes priority over whatever gMin/cellMinMax
		// said, at ANY mip -- not just the ones where this cell's own exit happens to be the
		// binding constraint. Descending to resolve a HIT/ambiguous call here used to shrink
		// cellExitT (a finer cell's own exit is always nearer than a coarser one's), which could
		// make exitingLayer read false again at the finer mip even though the coarser mip
		// correctly saw the layer edge as the real limit -- so a HIT/ambiguous result right at a
		// layer boundary would get "resolved" against a tiny, boundary-truncated cell instead of
		// ever reaching the next level's own slice, and a march that got stuck doing that one
		// finest-mip-sized step at a time could burn its whole iteration budget before actually
		// crossing the edge (TERRAIN_OCCLUSION_MAX_ITERATIONS is nowhere near big enough to crawl
		// a whole layer's width one mip-0 cell at a time). tEnd is already clamped to layerExitT
		// exactly whenever exitingLayer is true, at any mip, so jumping straight to L+1 here is
		// exact, not approximate.
		//
		// m = max(m - 1, -1) on the new level, not a fixed coarsest mip: texelSize doubles between
		// adjacent levels, so decrementing m by exactly 1 keeps cellSize = texelSize(L)*2^(m+1)
		// identical in world units across the boundary, whatever m happened to be. A fixed m=9
		// looks appealing ("jump to coarsest, don't bother special-casing") but is actually always
		// too coarse to use: layerHalfExtent(L) = 0.5*dim*texelSize(L)*0.9 (~460.8*texelSize(L) for
		// dim=1024) while a mip-9 cell is texelSize(L)*2^10 = 1024*texelSize(L) -- over twice the
		// *entire* level's own window, at every level, since both scale with texelSize(L) the same
		// way. Starting a level at m=9 means the very next sample's cellExitT is almost guaranteed
		// to exceed layerExitT again immediately, cascading through several levels in a few
		// iterations without ever actually sampling their real data -- exactly the kind of thing a
		// long, shallow ray (low sun angle, grazing a ridge) would hit hardest, since those are the
		// rays that need the most level transitions.
		if (exitingLayer) {
			// clearance stays at its sentinel here: this bypasses the clear/hit/ambiguous
			// classification entirely (that's the whole point -- see the comment above), so
			// there's no genuine margin to report either way.
			t = tEnd + kNudge;
			if (L + 1u >= ctx.numLevels) {
				t = maxDistance;
				return TERRAIN_MARCH_ADVANCE;
			}
			L += 1u;
			m = max(m - 1, -1);
			return TERRAIN_MARCH_ADVANCE;
		}

		if (gMin > cellMinMax.y + kEps) {
			// Clear: nothing in [t, tEnd] can occlude. Advance, then zoom out one mip.
			clearance = gMin - cellMinMax.y;
			t = tEnd + kNudge;
			m = min(m + 1, 9);
			return TERRAIN_MARCH_ADVANCE;
		}

		if (gMin < cellMinMax.x - kEps) {
			// A hit -- but confirm it at progressively finer mips rather than trusting
			// whichever coarse mip first flagged it, since a coarse cell's min can sit well
			// below the exact point the ray actually dips under. Can now descend to m=-1 (mip
			// 0, read-dilated -- sampleTerrainMinMaxMip's own comment) before giving up.
			if (m > -1) {
				m -= 1;
				continue;
			}
			return TERRAIN_MARCH_HIT;
		}

		// Ambiguous: this cell's bound straddles the ray's segment. Descend to resolve it.
		if (m > -1) {
			m -= 1;
			continue;
		}

		// Still ambiguous even at mip 0, the finest the chain offers -- the conservative
		// default is "not occluded here" (grazing a ridge must never register as a hit), so
		// advance past this cell rather than treat it as a hit. clearance stays at its sentinel:
		// this genuinely isn't a known-clear margin, and calculateTerrainSoftShadow must not
		// treat it as one (its own comment). Ascends afterward (m = min(m+1, 9), same as the
		// CLEAR path) rather than staying at the finest mip: an earlier version stayed at the
		// same resolution on the theory that nearby terrain is likely rough too, but a stretch
		// that's genuinely ambiguous across many consecutive mip-0 cells (plausible near rough
		// terrain, or near a layer edge where coarse-mip dilation makes a wide span look
		// ambiguous) would then crawl one mip-0-cell-width at a time with no way back to a
		// larger stride -- TERRAIN_OCCLUSION_MAX_ITERATIONS (64, shared across the whole trace,
		// every level) is nowhere near enough iterations to cross such a stretch that way, let
		// alone still reach a layer transition after it. Ascending costs nothing when the next,
		// larger cell turns out ambiguous too -- it just re-descends -- but lets a clear cell
		// immediately after a rough patch make real progress instead of continuing to crawl.
		t = tEnd + kNudge;
		m = min(m + 1, 9);
		return TERRAIN_MARCH_ADVANCE;
	}

	// Unreachable in practice (mip only spans -1..9) -- advance rather than loop forever.
	t = t1 + kNudge;
	return TERRAIN_MARCH_ADVANCE;
}

// Core occlusion routine: a hierarchical min-max march from origin along dir, up to
// maxDistance. On a hit, writes the hit distance into hitDistance and returns true. On a miss,
// leaves hitDistance untouched (so a caller's preset value survives) and returns false --
// including when the iteration budget runs out, which is always the safe (non-occluding)
// default for a culling test.
bool traceTerrainOcclusion(
	TerrainOcclusionContext ctx,
	vec3  origin,
	vec3  dir,
	float maxDistance,
	inout float hitDistance
) {
	if (ctx.minMaxIndex == 0u || ctx.numLevels == 0u || maxDistance <= 0.0) {
		return false;
	}

	vec3 d = normalize(dir);
	vec2 camXZ = uCameraPosition.xz;

	// g(t) = ray.y(t) + |ray.xz(t) - camXZ|^2 / (2R), expanded into C + B*t + A*t^2 once up
	// front -- cheap to evaluate/minimize over any [t0, t1] from here on (terrainRaySegmentMinG).
	vec2  o2 = origin.xz - camXZ;
	float A = dot(d.xz, d.xz) / (2.0 * FAKE_PLANET_RADIUS);
	float B = d.y + dot(o2, d.xz) / FAKE_PLANET_RADIUS;
	float C = origin.y + dot(o2, o2) / (2.0 * FAKE_PLANET_RADIUS);

	uint  L = min(terrainClipmapLevelFor(origin.xz, ctx.textureDim, ctx.numLevels), ctx.numLevels - 1u);
	int   m = 0;
	float t = 0.0;

	for (int iter = 0; iter < TERRAIN_OCCLUSION_MAX_ITERATIONS; ++iter) {
		if (t >= maxDistance) {
			return false;
		}

		float t0, t1, clearance;
		int   result = terrainMarchStep(ctx, origin, d, A, B, C, camXZ, maxDistance, L, m, t, t0, t1, clearance);
		if (result == TERRAIN_MARCH_HIT) {
			hitDistance = t1;
			return true;
		}
	}

	return false;
}

// Requested shape: vector + origin + distance. distance is in/out -- read as the max range to
// check within, overwritten with the hit distance only if occluded. origin/dir are render
// space, same as every other curved-world consumer's sample points.
bool isDirectionOccludedByTerrain(TerrainOcclusionContext ctx, vec3 origin, vec3 dir, inout float distance) {
	return traceTerrainOcclusion(ctx, origin, dir, distance, distance);
}

// Calculates smooth soft terrain shadow penumbra factor in [0.0, 1.0].
// 0.0 = fully shadowed (umbra), 1.0 = fully lit.
float calculateTerrainSoftShadow(
	TerrainOcclusionContext ctx,
	vec3  origin,
	vec3  dir,
	float maxDistance,
	float kPenumbra
) {
	if (ctx.minMaxIndex == 0u || ctx.numLevels == 0u || maxDistance <= 0.0) {
		return 1.0;
	}

	vec3 d = normalize(dir);
	vec2 camXZ = uCameraPosition.xz;

	vec2  o2 = origin.xz - camXZ;
	float A = dot(d.xz, d.xz) / (2.0 * FAKE_PLANET_RADIUS);
	float B = d.y + dot(o2, d.xz) / FAKE_PLANET_RADIUS;
	float C = origin.y + dot(o2, o2) / (2.0 * FAKE_PLANET_RADIUS);

	uint  L = min(terrainClipmapLevelFor(origin.xz, ctx.textureDim, ctx.numLevels), ctx.numLevels - 1u);
	int   m = 0;
	float t = 0.0;

	float softShadow = 1.0;

	for (int iter = 0; iter < TERRAIN_OCCLUSION_MAX_ITERATIONS; ++iter) {
		if (t >= maxDistance) {
			break;
		}

		float t0, t1, clearance;
		int   result = terrainMarchStep(ctx, origin, d, A, B, C, camXZ, maxDistance, L, m, t, t0, t1, clearance);
		if (result == TERRAIN_MARCH_HIT) {
			return 0.0;
		}

		// clearance < 0.0 is terrainMarchStep's sentinel for "no reliable margin" (an ambiguous
		// resolution, or a layer transition that bypassed classification entirely) -- previously
		// this was clamped to exactly 0.0 at the source, which made every one of those segments
		// read as "zero margin, so fully shadowed" (stepPenumbra = 0 -> softShadow = 0 ->
		// immediate return) despite the march's own stated policy being "not occluded here,"
		// not "fully occluded here." That silently turned every ambiguous cell and every level
		// transition into forced full shadow, which is backwards: uncertain should mean "this
		// segment contributes no information," not "this segment is definitely dark." Skipping
		// the update entirely when clearance is the sentinel lets only genuinely, confidently
		// clear segments narrow the penumbra.
		if (t1 > 0.2 && clearance >= 0.0) {
			// Evaluate continuous position at the center of the step
			float evalT = (t0 + t1) * 0.5;
			vec3 p = origin + d * evalT;

			// Sample the finest min-max bounds (mip 0) directly under the ray
			// to avoid blocky bounding box artifacts
			vec2 exactMinMax = sampleTerrainMinMaxMip(ctx.minMaxIndex, ctx.cubemapMinMaxIndex, p.xz, L, 0u, ctx.textureDim);

			float rayHeight = C + B * evalT + A * evalT * evalT;
			float trueClearance = rayHeight - exactMinMax.y;

			if (trueClearance >= 0.0) {
				float stepPenumbra = clamp(kPenumbra * trueClearance / evalT, 0.0, 1.0);
				softShadow = min(softShadow, stepPenumbra);
				if (softShadow <= 0.0) {
					return 0.0;
				}
			}
		}
	}

	return softShadow;
}

float calculateTerrainSoftShadow(
	TerrainOcclusionContext ctx,
	vec3  origin,
	vec3  dir,
	float maxDistance
) {
	return calculateTerrainSoftShadow(ctx, origin, dir, maxDistance, 16.0);
}

// Samples precalculated 3-layer 2D terrain shadow map (generated by terrain_shadow.comp).
// Returns soft shadow factor in [0.0, 1.0] (0.0 = fully shadowed, 1.0 = fully lit).
float evaluateTerrainShadowFromMap(
	uint shadowMapIndex,
	vec3 worldPos,
	vec3 lightDir,
	uint gridDim
) {
	if (shadowMapIndex == 0u || shadowMapIndex == 0xFFFFFFFFu) return 1.0;

	uint mapDim = (gridDim > 0u) ? gridDim : 1024u;

	// Pick cascade based on camera-centered distance
	vec2 offsetFromCam = worldPos.xz - uCameraPosition.xz;
	float distXZ = max(abs(offsetFromCam.x), abs(offsetFromCam.y));

	uint cascadeLODs[3] = uint[3](1u, 3u, 6u);
	uint cascade = 2u;

	float extent0 = float(mapDim) * texelSize(cascadeLODs[0]);
	float extent1 = float(mapDim) * texelSize(cascadeLODs[1]);

	if (distXZ <= extent0 * 0.5) {
		cascade = 0u;
	} else if (distXZ <= extent1 * 0.5) {
		cascade = 1u;
	} else {
		cascade = 2u;
	}

	float cascadeExtent = float(mapDim) * texelSize(cascadeLODs[cascade]);
	vec2 shadowUV = (offsetFromCam / cascadeExtent) + 0.5;

	if (any(lessThan(shadowUV, vec2(0.0))) || any(greaterThanEqual(shadowUV, vec2(1.0)))) {
		return 1.0;
	}

	vec4 shadowData = SAMPLE_ARRAY_CLAMP(shadowMapIndex, vec3(shadowUV, float(cascade)));

	float terrainHeight = shadowData.r;
	float occlusionDepth = shadowData.g;
	float groundShadow = shadowData.a;

	if (groundShadow >= 0.99) return 1.0;

	float dropOff = terrainDropOff(worldPos.xz);
	float renderTerrainH = terrainHeight - dropOff;

	float heightAboveGround = worldPos.y - renderTerrainH;

	// Ground surface and points below occlusion depth remain shadowed
	if (occlusionDepth <= 0.001 || heightAboveGround <= 0.0) {
		return groundShadow;
	}

	if (heightAboveGround >= occlusionDepth) {
		return 1.0;
	}

	float clearFraction = clamp(heightAboveGround / occlusionDepth, 0.0, 1.0);
	return mix(groundShadow, 1.0, clearFraction);
}

// Shared by isAABBOccludedByTerrain's three rays: traces from `from` toward `target` (both
// flat-world positions already curvature-dropped by the caller), capped just short of the
// target so terrain *at* the target can't flag it as self-occluded.
bool terrainOcclusionRayHitsTarget(TerrainOcclusionContext ctx, vec3 from, vec3 target) {
	vec3  toTarget = target - from;
	float dist = length(toTarget);
	if (dist <= 0.5) {
		return false;
	}
	float hitDist = dist - 0.5;
	return traceTerrainOcclusion(ctx, from, toTarget, hitDist, hitDist);
}

// Requested shape: world-space AABB. Cheap horizon full-reject first, then a 3-ray silhouette
// set from the camera: the box's two azimuth-extreme top corners (as seen from the camera) plus
// the top face's highest-elevation point once curvature is applied. All three must hit for the
// box to count as occluded -- closing the gap a 2-ray (nearest/center) test leaves at a box's
// edges, where it can poke out past a ridge between the two sampled points without either ray
// detecting it.
bool isAABBOccludedByTerrain(TerrainOcclusionContext ctx, vec3 aabbMin, vec3 aabbMax, float horizonSlack) {
	vec3 camPos = uCameraPosition.xyz;

	float margin = texelSize(0u);
	if (camPos.x >= aabbMin.x - margin && camPos.x <= aabbMax.x + margin &&
	    camPos.z >= aabbMin.z - margin && camPos.z <= aabbMax.z + margin) {
		return false; // camera is standing in/on this footprint
	}

	vec2  nearestXZ = clamp(camPos.xz, aabbMin.xz, aabbMax.xz);
	float horizDist = length(nearestXZ - camPos.xz);
	float camAlt = camPos.y - ctx.refHeight;
	float boxAlt = aabbMax.y - ctx.refHeight;
	if (isBeyondHorizon(horizDist, boxAlt, camAlt, horizonSlack)) {
		return true;
	}

	// Azimuth-extreme top corners, found by comparing atan2 angles of all 4 relative to the
	// camera -- exact for the (overwhelmingly common) case where the box doesn't straddle the
	// camera's +/-180 degree branch cut; a box that rare/degenerate case picks a merely
	// suboptimal pair of corners, not an incorrect result (the box still needs all 3 rays to
	// hit before it's culled).
	vec2 c00 = vec2(aabbMin.x, aabbMin.z) - camPos.xz;
	vec2 c10 = vec2(aabbMax.x, aabbMin.z) - camPos.xz;
	vec2 c01 = vec2(aabbMin.x, aabbMax.z) - camPos.xz;
	vec2 c11 = vec2(aabbMax.x, aabbMax.z) - camPos.xz;
	float a00 = atan(c00.y, c00.x);
	float a10 = atan(c10.y, c10.x);
	float a01 = atan(c01.y, c01.x);
	float a11 = atan(c11.y, c11.x);

	vec2  silMin = c00;
	float aMin = a00;
	if (a10 < aMin) { aMin = a10; silMin = c10; }
	if (a01 < aMin) { aMin = a01; silMin = c01; }
	if (a11 < aMin) { aMin = a11; silMin = c11; }

	vec2  silMax = c00;
	float aMax = a00;
	if (a10 > aMax) { aMax = a10; silMax = c10; }
	if (a01 > aMax) { aMax = a01; silMax = c01; }
	if (a11 > aMax) { aMax = a11; silMax = c11; }

	// Highest-elevation top-face point: the horizontal distance r that maximizes
	// (topY - camY)/r - r/(2R) is r = sqrt(2R*(camY - topY)) when the box top sits below the
	// camera (closer is otherwise always "higher", so r -> 0, i.e. the nearest point), clamped
	// to where the camera-to-center ray actually crosses the box's footprint.
	vec2  centerXZ = 0.5 * (aabbMin.xz + aabbMax.xz);
	vec2  toCenterXZ = centerXZ - camPos.xz;
	float distToCenter = length(toCenterXZ);
	vec2  dirToCenterXZ = (distToCenter > 1e-5) ? toCenterXZ / distToCenter : vec2(1.0, 0.0);

	vec2 invDir = 1.0 / dirToCenterXZ;
	vec2 tSlab0 = (aabbMin.xz - camPos.xz) * invDir;
	vec2 tSlab1 = (aabbMax.xz - camPos.xz) * invDir;
	vec2 tSlabMin = min(tSlab0, tSlab1);
	vec2 tSlabMax = max(tSlab0, tSlab1);
	float rNear = max(0.0, max(tSlabMin.x, tSlabMin.y));
	float rFar = max(rNear, min(tSlabMax.x, tSlabMax.y));

	float camAltAboveTop = camPos.y - aabbMax.y;
	float rPeak = (camAltAboveTop > 0.0) ? sqrt(2.0 * FAKE_PLANET_RADIUS * camAltAboveTop) : 0.0;
	float rHigh = clamp(rPeak, rNear, rFar);
	vec2  highXZ = camPos.xz + dirToCenterXZ * rHigh;

	vec3 highPoint = vec3(highXZ.x, aabbMax.y, highXZ.y);
	vec3 corner1 = vec3(camPos.x + silMin.x, aabbMax.y, camPos.z + silMin.y);
	vec3 corner2 = vec3(camPos.x + silMax.x, aabbMax.y, camPos.z + silMax.y);

	highPoint.y -= terrainDropOff(highPoint.xz);
	corner1.y -= terrainDropOff(corner1.xz);
	corner2.y -= terrainDropOff(corner2.xz);

	// Highest-elevation point first -- the most likely of the three to miss, so a visible box
	// fails fast instead of paying for two more traces first.
	if (!terrainOcclusionRayHitsTarget(ctx, camPos, highPoint)) {
		return false;
	}
	if (!terrainOcclusionRayHitsTarget(ctx, camPos, corner1)) {
		return false;
	}
	return terrainOcclusionRayHitsTarget(ctx, camPos, corner2);
}

// -----------------------------------------------------------------------------
// EROSION FILTER
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// COLOR MAPPING
// -----------------------------------------------------------------------------

/**
 * Apply realistic erosion color mapping.
 *
 * @param albedo Base terrain color.
 * @param ridgeMap Ridge map output from ErosionFilter (-1 to 1).
 * @param heightDelta Height change from erosion.
 * @param sedimentColor Color of accumulated sediment in creases.
 * @param ridgeColor Color of exposed rock on ridges.
 */
vec3 applyErosionColorMapping(vec3 albedo, float ridgeMap, float heightDelta, vec3 sedimentColor, vec3 ridgeColor) {
	// Darken/Color creases (sediment/water accumulation)
	float creaseMask = 1.0 - smoothstep(-0.8, 0.2, ridgeMap);
	albedo = mix(albedo, sedimentColor, creaseMask * 0.5);

	// Highlight ridges (exposed rock/weathering)
	float ridgeMask = smoothstep(0.2, 0.8, ridgeMap);
	albedo = mix(albedo, ridgeColor, ridgeMask * 0.3);

	// Subtle darkening in deeper eroded areas
	albedo *= (1.0 - clamp01(-heightDelta * 2.0) * 0.2);

	return albedo;
}

/**
 * Default color mapping with sensible defaults.
 */
vec3 applyErosionColorMappingDefault(vec3 albedo, float ridgeMap, float heightDelta) {
	vec3 sediment = vec3(0.1, 0.08, 0.05); // Dark dirt/moist soil
	vec3 rock = vec3(0.8, 0.75, 0.7);      // Lighter exposed rock
	return applyErosionColorMapping(albedo, ridgeMap, heightDelta, sediment, rock);
}

/**
 * Returns regional rock color mapped from geological biome parameter (0.0 to 1.0).
 */
vec3 getGeologicalRockColor(float geoBiome) {
	vec3 chalk = vec3(0.55, 0.52, 0.47);        // Pale limestone / chalk / light clay
	vec3 redSandstone = vec3(0.58, 0.28, 0.20); // Warm iron-rich red sandstone
	vec3 granite = vec3(0.35, 0.36, 0.38);      // Cool neutral grey granite / slate
	vec3 basalt = vec3(0.20, 0.19, 0.22);       // Dark volcanic basalt / igneous rock

	if (geoBiome < 0.33) {
		return mix(chalk, redSandstone, geoBiome / 0.33);
	} else if (geoBiome < 0.66) {
		return mix(redSandstone, granite, (geoBiome - 0.33) / 0.33);
	} else {
		return mix(granite, basalt, clamp((geoBiome - 0.66) / 0.34, 0.0, 1.0));
	}
}


#endif // BRASSICA_TERRAIN_GLSL
