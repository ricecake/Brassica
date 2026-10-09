#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"
#include "helpers/culling.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/whittaker.glsl"

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
// A flat ray march against the min/max mip chain -- a stepping stone toward a real
// hierarchical HiZ structure, not that structure itself. One shared core
// (traceTerrainOcclusion) plus two thin wrappers matching the two shapes callers need:
// an AABB test and a direct origin/direction/distance test. Horizon-drop aware: every
// sample applies the same worldY -= d^2/(2*FAKE_PLANET_RADIUS) correction every other
// curved-world consumer uses (terrain.mesh/foliage.mesh/particle.mesh/deferred.frag), so
// terrain that has genuinely curved below the horizon is never mistaken for an occluder --
// isBeyondHorizon (helpers/culling.glsl) is used as a cheap full-reject in the AABB
// wrapper before any marching happens.

#ifndef TERRAIN_OCCLUSION_MAX_STEPS
#define TERRAIN_OCCLUSION_MAX_STEPS 48
#endif
#ifndef TERRAIN_OCCLUSION_STEP_TEXELS
#define TERRAIN_OCCLUSION_STEP_TEXELS 4.0
#endif

// Bundles the four indices/counts every occlusion call needs so call sites pass one value
// instead of five. refHeight is the reference surface isBeyondHorizon measures altitude
// from -- pass the water level (or 0 on dry maps).
struct TerrainOcclusionContext {
	uint  clipmapIndex;
	uint  minMaxIndex;
	uint  textureDim;
	uint  numLevels;
	float refHeight;
};

TerrainOcclusionContext makeTerrainOcclusionContext(
	uint  clipmapIndex,
	uint  minMaxIndex,
	uint  textureDim,
	uint  numLevels,
	float refHeight
) {
	TerrainOcclusionContext ctx;
	ctx.clipmapIndex = clipmapIndex;
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

// Camera-centered LOD pick for a world-space point, generalizing deferred.frag's
// calculateRayLOD (which hardcodes baseRadius=272.0 for its own short reflection-probe
// range) into something usable for an arbitrarily long occlusion march. Returns
// numLevels itself as an "out of clipmap coverage" sentinel for the march to stop on.
uint terrainClipmapLevelFor(vec2 worldXZ, uint textureDim, uint numLevels) {
	const float kClipmapCoverageSafety = 0.9; // leaves slack for snapping/update lag
	float dim = float((textureDim > 0u) ? textureDim : 1024u);
	float halfExtent0 = 0.5 * dim * texelSize(0u) * kClipmapCoverageSafety;

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

// Core occlusion routine: a flat march from origin along dir, up to maxDistance. On a
// hit, writes the hit distance into hitDistance and returns true. On a miss, leaves
// hitDistance untouched (so a caller's preset value survives) and returns false.
bool traceTerrainOcclusion(
	TerrainOcclusionContext ctx,
	vec3  origin,
	vec3  dir,
	float maxDistance,
	inout float hitDistance
) {
	if (ctx.clipmapIndex == 0u || ctx.minMaxIndex == 0u || ctx.numLevels == 0u || maxDistance <= 0.0) {
		return false;
	}

	vec3 d = normalize(dir);

	uint  originLevel = min(terrainClipmapLevelFor(origin.xz, ctx.textureDim, ctx.numLevels), ctx.numLevels - 1u);
	float t = max(0.25, 0.5 * texelSize(originLevel));

	// Coarsest-mip ceiling at the camera -- an upper bound on the whole clipmap's height.
	// Once the ray is above it and still pulling away relative to the curved surface,
	// nothing further out can occlude.
	vec2 camMinMax = sampleTerrainMinMaxMip(ctx.minMaxIndex, uCameraPosition.xz, ctx.numLevels - 1u, 10u, ctx.textureDim);

	for (int i = 0; i < TERRAIN_OCCLUSION_MAX_STEPS; ++i) {
		if (t >= maxDistance) {
			break;
		}

		vec3 p = origin + d * t;
		uint level = terrainClipmapLevelFor(p.xz, ctx.textureDim, ctx.numLevels);
		if (level >= ctx.numLevels) {
			break; // out of clipmap coverage, nothing further to test against
		}

		float dropP = terrainDropOff(p.xz);
		if (p.y > camMinMax.y - dropP + 50.0 && d.y >= 0.0) {
			break;
		}

		float ts = texelSize(level);
		float stepLen = clamp(ts * TERRAIN_OCCLUSION_STEP_TEXELS, 0.5, 64.0);
		float tNext = min(t + stepLen, maxDistance);
		vec3  pEnd = origin + d * tNext;
		float dropEnd = terrainDropOff(pEnd.xz);

		// Cheap skip: compare against the loosest (max) bound over this step's footprint
		// at a mip coarse enough to cover it, before falling through to an exact sample.
		uint mip = uint(clamp(ceil(log2(max(tNext - t, 1.0) / max(ts, 0.01))), 0.0, 10.0));
		vec2 minMaxStart = sampleTerrainMinMaxMip(ctx.minMaxIndex, p.xz, level, mip, ctx.textureDim);
		vec2 minMaxEnd = sampleTerrainMinMaxMip(ctx.minMaxIndex, pEnd.xz, level, mip, ctx.textureDim);
		float segMaxHeight = max(minMaxStart.y, minMaxEnd.y);
		float minDrop = min(dropP, dropEnd);
		float rayMinY = min(p.y, pEnd.y);

		if (rayMinY > segMaxHeight - minDrop + 0.5) {
			t = tNext;
			continue;
		}

		float exactHeight = sampleTerrainClipmap(ctx.clipmapIndex, pEnd.xz, level, ctx.textureDim).r - dropEnd;
		if (pEnd.y <= exactHeight) {
			float lo = t, hi = tNext;
			for (int b = 0; b < 3; ++b) {
				float mid = 0.5 * (lo + hi);
				vec3  pm = origin + d * mid;
				uint  lvlM = min(terrainClipmapLevelFor(pm.xz, ctx.textureDim, ctx.numLevels), ctx.numLevels - 1u);
				float hM =
					sampleTerrainClipmap(ctx.clipmapIndex, pm.xz, lvlM, ctx.textureDim).r - terrainDropOff(pm.xz);
				if (pm.y <= hM) {
					hi = mid;
				} else {
					lo = mid;
				}
			}
			hitDistance = hi;
			return true;
		}

		t = tNext;
	}

	return false;
}

// Requested shape: vector + origin + distance. distance is in/out -- read as the max range
// to check within, overwritten with the hit distance only if occluded.
bool isDirectionOccludedByTerrain(TerrainOcclusionContext ctx, vec3 origin, vec3 dir, inout float distance) {
	return traceTerrainOcclusion(ctx, origin, dir, distance, distance);
}

// Requested shape: world-space AABB. Cheap horizon full-reject first, then two rays from
// the camera at the box's nearest-to-camera and center top-face points, each capped just
// short of its own box entry so terrain *inside* the box can't flag the box as
// self-occluded. Occluded only if both rays hit something before reaching the box.
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

	vec3 nearestTop = vec3(nearestXZ.x, aabbMax.y, nearestXZ.y);
	vec3 centerTop = vec3(0.5 * (aabbMin.x + aabbMax.x), aabbMax.y, 0.5 * (aabbMin.z + aabbMax.z));

	vec3  toNearest = nearestTop - camPos;
	float distNearest = length(toNearest);
	if (distNearest <= 0.5) {
		return false;
	}
	float hitNearest = distNearest - 0.5;
	if (!traceTerrainOcclusion(ctx, camPos, toNearest, hitNearest, hitNearest)) {
		return false;
	}

	vec3  toCenter = centerTop - camPos;
	float distCenter = length(toCenter);
	if (distCenter <= 0.5) {
		return false;
	}
	float hitCenter = distCenter - 0.5;
	return traceTerrainOcclusion(ctx, camPos, toCenter, hitCenter, hitCenter);
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
