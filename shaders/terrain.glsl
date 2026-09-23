#ifndef BRASSICA_TERRAIN_GLSL
#define BRASSICA_TERRAIN_GLSL

#include "bindless.glsl"
#include "common.glsl"
#include "lygia/generative/psrdnoise.glsl"

void eval_terrain(vec3 p, TerrainConfig config, out float out_height, out vec3 out_normal, out vec3 out_grad) {
	p *= config.spatial_scale;

	float continent_mask;
	vec3 grad_continent_mask;
	evaluate_soft_voronoi_pseudosphere2(p, 32, continent_mask, grad_continent_mask);

	vec3 grad_noise;
	float warp = psrdnoise(p*continent_mask, vec3(0), 0.0);

	float noise_height = dot_noise_fbm(p+warp, 4, 0.0, grad_noise);

	vec3 grad_noise2;
	float warp2 = psrdnoise(p/continent_mask, vec3(0), 10.0);
	float noise_height2 = dot_noise_fbm(p*warp2, 4, 3.0, grad_noise2);

	grad_noise += grad_noise2;
	noise_height += noise_height2;

	float final_height = noise_height * continent_mask;
	vec3 final_grad = (grad_noise * continent_mask) + (noise_height * grad_continent_mask);

	float height_amplitude = config.max_height - config.min_height;
	vec3 scaled_grad = final_grad * height_amplitude * config.spatial_scale;

	out_normal = normalize(vec3(-scaled_grad.x, 1.0, -scaled_grad.z));
	out_height = remap(final_height, 0.0, 1.0, config.min_height, config.max_height);
	out_grad = scaled_grad;
}

float getLODScale(float lod) {
	if (lod <= 3.0) {
		return pow(2.0, lod);
	} else {
		return 8.0 * pow(2.25, lod - 3.0);
	}
}

float texelSize(
	uint  level
) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	return baseTexelSize * getLODScale(float(level));
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

vec4 sampleTerrainBiome(uint biomeIndex, vec2 worldXZ, uint level) {
	return sampleTerrainBiome(biomeIndex, worldXZ, level, 1088u, uvec4(0u), uvec4(0u), uvec4(0u));
}

// Sample terrain tile visibility map (r = visibility flag / occlusion factor)
float sampleTerrainTileVisibility(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7,
	uvec4 lodOffsets8_11
) {
	vec2 uv = sampleToroidalUV(worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, lodOffsets8_11);
	return SAMPLE_ARRAY_WRAP(visIndex, vec3(uv, float(level))).r;
}

float sampleTerrainTileVisibility(
	uint  visIndex,
	vec2  worldXZ,
	uint  level,
	uint  textureDim,
	uvec4 lodOffsets0_3,
	uvec4 lodOffsets4_7
) {
	return sampleTerrainTileVisibility(visIndex, worldXZ, level, textureDim, lodOffsets0_3, lodOffsets4_7, uvec4(0u));
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
