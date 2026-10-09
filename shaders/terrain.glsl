#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"
#include "helpers/culling.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/whittaker.glsl"
#include "terrain_dirty.glsl"

// Toroidal UV mapping helper for terrain clipmap textures
vec2 sampleToroidalUV(vec2 worldXZ, uint level, uint textureDim) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float texelSize = baseTexelSize * getLODScale(float(level));
	float dim = float((textureDim > 0u) ? textureDim : 1024u);

	// Find the discrete world grid coordinate
	vec2 worldGrid = floor(worldXZ / texelSize);

	// Map directly to the wrapped texture coordinate.
	// The dim * 0.5 shift maintains the camera-centered local window.
	vec2 texelCoord = mod(worldGrid + dim * 0.5, dim);

	return texelCoord / dim;
}

// Sample terrain height (r) and normal (gba) from clipmap
vec4 sampleTerrainClipmap(
	uint  clipmapIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim);
	return SAMPLE_ARRAY_WRAP(clipmapIndex, vec3(uv, float(level)));
}

// Sample terrain min-max height map (r = min height, g = max height)
vec2 sampleTerrainMinMax(
	uint  minMaxIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim);
	return SAMPLE_ARRAY_WRAP(minMaxIndex, vec3(uv, float(level))).rg;
}

// Sample terrain biome map (r = biome weight/type, g = terrain variance for LOD adjustments, b = detail, a = moisture)
vec4 sampleTerrainBiome(
	uint  biomeIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim);
	return SAMPLE_ARRAY_WRAP(biomeIndex, vec3(uv, float(level)));
}

// Sample octahedral weather and Whittaker biome texture
vec4 sampleTerrainWeatherBiome(uint weatherStorageIdx, vec3 worldPos) {
	vec3 dir = normalize(worldPos - vec3(0.0, -FAKE_PLANET_RADIUS, 0.0));
	vec2 uv = directionToOctahedralUV(dir);
	return SAMPLE_LINEAR(weatherStorageIdx, uv);
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
	uint  textureDim;
	uint  numLevels;
	float refHeight;
};

TerrainOcclusionContext makeTerrainOcclusionContext(
	uint  minMaxIndex,
	uint  textureDim,
	uint  numLevels,
	float refHeight
) {
	TerrainOcclusionContext ctx;
	ctx.minMaxIndex = minMaxIndex;
	ctx.textureDim = textureDim;
	ctx.numLevels = numLevels;
	ctx.refHeight = refHeight;
	return ctx;
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
	vec2  worldXZ,
	uint  level,
	uint  mip,
	uint  textureDim
) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float ts = baseTexelSize * getLODScale(float(level));
	float dim = float((textureDim > 0u) ? textureDim : 1024u);

	vec2 worldGrid = floor(worldXZ / ts);
	vec2 texelCoord = mod(worldGrid + dim * 0.5, dim);

	// TerrainMinMaxDesc fixes mips=11 (0..10) for a 1024-wide map -- same coupling
	// TerrainMinMaxDownsampleNode's kOutputMips already hardcodes.
	int  mipDim = max(1, int(dim) >> int(mip));
	ivec2 mipTexel = clamp(ivec2(texelCoord) >> int(mip), ivec2(0), ivec2(mipDim - 1));

	return texelFetch(
		sampler2DArray(uTextureArrays[nonuniformEXT(minMaxIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		ivec3(mipTexel, int(level)),
		int(mip)
	).rg;
}

// Half-extent (world units) of level L's 0.9-safety Chebyshev window around the camera --
// shared with the hierarchical tracer below, which needs every level's window, not just
// level 0's.
float terrainLevelHalfExtent(uint level, uint textureDim) {
	const float kClipmapCoverageSafety = 0.9; // leaves slack for snapping/update lag
	float dim = float((textureDim > 0u) ? textureDim : 1024u);
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

// The conservative min/max bound for the clipmap cell (level, mip) containing worldXZ. Mip 0
// itself is never dilated -- terrain_gen.comp writes it as a raw per-texel height, min == max --
// so a mip-0 cell only bounds its own corner sample, not the interpolated surface
// sampleTerrainClipmap actually draws across the cell's footprint. Dilate it manually here the
// same way SpdLoadSourceImage dilates every coarser mip during the SPD build
// (TerrainMinMaxDownsampleNode): min/max over the cell's own corner plus its three neighbors one
// texel over. Mip 1+ reads its already-dilated value directly, in one fetch.
vec2 terrainCellMinMax(TerrainOcclusionContext ctx, vec2 worldXZ, uint level, uint mip) {
	if (mip > 0u) {
		return sampleTerrainMinMaxMip(ctx.minMaxIndex, worldXZ, level, mip, ctx.textureDim);
	}
	float ts = texelSize(level);
	vec2 a = sampleTerrainMinMaxMip(ctx.minMaxIndex, worldXZ, level, 0u, ctx.textureDim);
	vec2 b = sampleTerrainMinMaxMip(ctx.minMaxIndex, worldXZ + vec2(ts, 0.0), level, 0u, ctx.textureDim);
	vec2 c = sampleTerrainMinMaxMip(ctx.minMaxIndex, worldXZ + vec2(0.0, ts), level, 0u, ctx.textureDim);
	vec2 e = sampleTerrainMinMaxMip(ctx.minMaxIndex, worldXZ + vec2(ts, ts), level, 0u, ctx.textureDim);
	return vec2(min(min(a.x, b.x), min(c.x, e.x)), max(max(a.y, b.y), max(c.y, e.y)));
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
// at parameter t, exactly -- see traceTerrainOcclusion for the A/B/C derivation. Convex since
// A = |dir.xz|^2 / (2*FAKE_PLANET_RADIUS) >= 0, so its minimum over [t0, t1] is at the vertex
// clamped into the interval (or, in the degenerate near-vertical/A~0 case, at whichever endpoint
// is lower -- g is then effectively linear in t).
float terrainRaySegmentMinG(float A, float B, float C, float t0, float t1) {
	float tStar = (A > 1e-12) ? clamp(-B / (2.0 * A), t0, t1) : (B >= 0.0 ? t0 : t1);
	return C + B * tStar + A * tStar * tStar;
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

	uint L = min(terrainClipmapLevelFor(origin.xz, ctx.textureDim, ctx.numLevels), ctx.numLevels - 1u);
	int  m = 0;
	float t = 0.0;

	const float kEps = 0.1;     // bias toward "ambiguous" (resolved by descending) over a wrong call
	const float kNudge = 1e-3;  // push just past a cell/layer boundary so the next floor() is unambiguous

	for (int iter = 0; iter < TERRAIN_OCCLUSION_MAX_ITERATIONS; ++iter) {
		if (t >= maxDistance) {
			return false;
		}

		float tsL = texelSize(L);
		float cellSize = tsL * exp2(float(m));
		vec3  p = origin + d * t;

		float cellExitT = terrainCellExitT(origin, d, p.xz, cellSize);
		float layerHalfExtent = terrainLevelHalfExtent(L, ctx.textureDim);
		float layerExitT = terrainLayerExitT(origin, d, camXZ, layerHalfExtent);
		bool  exitingLayer = layerExitT <= cellExitT;
		float t1 = min(min(cellExitT, layerExitT), maxDistance);
		if (t1 <= t) {
			t1 = t + cellSize * 1e-4; // degenerate (e.g. dir parallel to a cell edge through its corner)
		}

		vec2  cellMinMax = terrainCellMinMax(ctx, p.xz, L, uint(m));
		float gMin = terrainRaySegmentMinG(A, B, C, t, t1);

		if (gMin > cellMinMax.y + kEps) {
			// Clear: nothing in [t, t1] can occlude. Advance, then zoom out one mip -- unless
			// that would exit this layer's own safety window, in which case move out to the
			// next (coarser) layer one mip finer than its own 0, keeping the world cell size
			// roughly continuous across the layer boundary instead of jumping.
			t = t1 + kNudge;
			if (exitingLayer) {
				if (L + 1u >= ctx.numLevels) {
					return false;
				}
				L += 1u;
				m = max(m - 1, 0);
			} else {
				m = min(m + 1, 10);
			}
			continue;
		}

		if (gMin < cellMinMax.x - kEps) {
			// A hit -- but confirm it at progressively finer mips rather than trusting
			// whichever coarse mip first flagged it, since a coarse cell's min can sit well
			// below the exact point the ray actually dips under.
			if (m > 0) {
				m -= 1;
				continue;
			}
			hitDistance = t1;
			return true;
		}

		// Ambiguous: this cell's bound straddles the ray's segment. Descend to resolve it.
		if (m > 0) {
			m -= 1;
			continue;
		}

		// Still ambiguous at mip 0, the finest level the dilated chain offers -- the
		// conservative default is "not occluded here" (grazing a ridge must never register as
		// a hit), so advance past this cell at the same resolution instead of ascending.
		t = t1 + kNudge;
		if (exitingLayer) {
			if (L + 1u >= ctx.numLevels) {
				return false;
			}
			L += 1u;
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

/**
 * Advanced Terrain Erosion Filter
 *
 * @param p World or UV coordinates for evaluation.
 * @param heightAndSlope Input height (x) and derivative (yz).
 * @param fadeTarget Value to fade towards (-1 at valleys, 1 at peaks).
 * @param strength Overall erosion strength.
 * @param gullyWeight Balance between sharpening and gullies (0 to 1).
 * @param detail Detail restriction (higher = more detail on shallow slopes).
 * @param rounding Rounding parameters (x: ridge, y: crease, z: input mult, w: octave mult).
 * @param onset Controls where erosion takes effect (x: initial, y: octave, z: ridgemap initial, w: ridgemap octave).
 * @param assumedSlope Override slope for initial gully direction (x: value, y: override amount).
 * @param scale Spatial scale of the erosion.
 * @param octaves Number of gully scales to layer.
 * @param lacunarity Frequency multiplier per octave.
 * @param gain Magnitude multiplier per octave.
 * @param cellScale Size of Phacelle noise cells.
 * @param normalization Consistency of gully magnitudes.
 * @param ridgeMap Output: -1 on creases, 1 on ridges.
 * @param substrate Output: -1 for heavy erosion (valleys), 1 for deposition (plains/ridges).
 * @param debug Output: for visualization.
 * @return vec4(heightDelta, slopeDelta.xy, totalMagnitude)
 */
vec4 ErosionFilter(
	in vec2   p,
	vec3      heightAndSlope,
	float     fadeTarget,
	float     strength,
	float     gullyWeight,
	float     detail,
	vec4      rounding,
	vec4      onset,
	vec2      assumedSlope,
	float     scale,
	int       octaves,
	float     lacunarity,
	float     gain,
	float     cellScale,
	float     normalization,
	out float ridgeMap,
	out float substrate,
	out float debug
) {
	strength *= (scale + 10.0);
	fadeTarget = clamp(fadeTarget, -1.0, 1.0);

	scale *= 100.0;

	vec3  inputHeightAndSlope = heightAndSlope;
	float freq = 1.0 / (scale * cellScale);
	float slopeLength = max(length(heightAndSlope.yz), 1e-10);
	float magnitude = 0.0;
	float roundingMult = 1.0;

	float roundingForInput = mix(rounding.y, rounding.x, clamp01(fadeTarget + 0.5)) * rounding.z;
	float combiMask = ease_out(smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

	float ridgeMapCombiMask = ease_out(slopeLength * onset.z);
	float ridgeMapFadeTarget = fadeTarget;

	vec2 gullySlope = mix(heightAndSlope.yz, heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);

	for (int i = 0; i < octaves; i++) {
		vec4 phacelle = PhacelleNoise(p * freq, safe_normalize(gullySlope), cellScale, 0.25, normalization);
		phacelle.zw *= -freq;
		float sloping = abs(phacelle.y);

		gullySlope += sign(phacelle.y) * phacelle.zw * strength * gullyWeight;

		vec3 gullies = vec3(phacelle.x, phacelle.y * phacelle.zw);
		vec3 fadedGullies = mix(vec3(fadeTarget, 0.0, 0.0), gullies * gullyWeight, combiMask);
		heightAndSlope += fadedGullies * strength;
		magnitude += strength;

		fadeTarget = fadedGullies.x;

		float roundingForOctave = mix(rounding.y, rounding.x, clamp01(phacelle.x + 0.5)) * roundingMult;
		float newMask = ease_out(smooth_start(sloping * onset.y, roundingForOctave * onset.y));
		combiMask = pow_inv(combiMask, detail) * newMask;

		ridgeMapFadeTarget = mix(ridgeMapFadeTarget, gullies.x, ridgeMapCombiMask);
		float newRidgeMapMask = ease_out(sloping * onset.w);
		ridgeMapCombiMask = ridgeMapCombiMask * newRidgeMapMask;

		strength *= gain;
		freq *= lacunarity;
		roundingMult *= rounding.w;
	}

	ridgeMap = ridgeMapFadeTarget * (1.0 - ridgeMapCombiMask);
	substrate = clamp(fadeTarget, -1.0, 1.0);
	debug = fadeTarget;

	vec3 heightAndSlopeDelta = heightAndSlope - inputHeightAndSlope;
	return vec4(heightAndSlopeDelta, magnitude);
}

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


#endif // BRASSICA_TERRAIN_GLSL
