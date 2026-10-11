#ifndef BRASSICA_SCREEN_SPACE_SHADOWS_GLSL
#define BRASSICA_SCREEN_SPACE_SHADOWS_GLSL

#include "bindless.glsl"

/**
 * Calculates screen space shadows using HiZ depth pyramid hierarchy.
 *
 * fragPos: Absolute world position of the fragment.
 * normal: Normal at the fragment.
 * lightDir: Direction vector pointing TOWARDS the light source.
 * hizIndex: Bindless sampled texture index for HiZTexture.
 * depthIndex: Bindless sampled texture index for GBufferDepth.
 *
 * Returns shadow visibility factor [0.0 = fully shadowed, 1.0 = fully lit].
 */
float calculateScreenSpaceShadowHiZ(
	vec3 fragPos,
	vec3 normal,
	vec3 lightDir,
	uint hizIndex,
	uint depthIndex
) {
	if (hizIndex == 0u || depthIndex == 0u) {
		return 1.0;
	}

	// Offset ray start along normal and light direction to prevent self-shadowing acne
	vec3 rayOrigin = fragPos + normal * 0.08 + lightDir * 0.08;
	vec3 rayDir = normalize(lightDir);

	float maxRayDist = 40.0; // Maximum world-space shadow search distance (meters)
	const int maxSteps = 32;
	float stepDist = 0.5;    // Base world step distance

	float currentDist = stepDist;
	float shadowVisibility = 1.0;

	// Query total available mips in HiZ texture
	int maxMip = textureQueryLevels(
		sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP])
	) - 1;

	for (int i = 0; i < maxSteps && currentDist < maxRayDist; ++i) {
		vec3 currentPos = rayOrigin + rayDir * currentDist;

		// Project current position into screen clip space
		vec4 clipPos = uViewProjMatrix * vec4(currentPos, 1.0);

		if (clipPos.w <= 0.001) {
			break; // Ray went behind camera
		}

		vec3 ndc = clipPos.xyz / clipPos.w;
		vec2 uv = ndc.xy * 0.5 + 0.5;

		// Check screen bounds
		if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
			break;
		}

		float rayDepth = ndc.z;
		if (rayDepth < 0.0 || rayDepth > 1.0) {
			break;
		}

		// Select HiZ mip level based on step distance
		float mipLevel = clamp(floor(log2(max(currentDist, 1.0))), 0.0, float(max(maxMip, 0)));

		// Sample HiZ min/max depth bounds at mip level
		vec2 hizMinMax = textureLod(
			sampler2D(uTextures2D[nonuniformEXT(hizIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
			uv,
			mipLevel
		).rg;

		float minDepth = hizMinMax.r; // Closest surface in HiZ cell (Standard Z: smaller = near)
		float maxDepth = hizMinMax.g; // Furthest surface in HiZ cell

		// Standard Z: if ray depth is closer to camera than cell's minimum surface depth,
		// the ray cannot hit any surface in this cell.
		if (rayDepth < minDepth - 0.0001) {
			// Take an accelerated step scaled by the HiZ hierarchy level
			float stepMult = max(1.0, exp2(mipLevel * 0.5));
			currentDist += stepDist * stepMult;
			continue;
		}

		// Ray is near or past cell min depth: test precise scene depth at mip 0
		float sceneDepth = SAMPLE_NEAREST(depthIndex, uv).r;

		const float depthBias = 0.0002;
		const float maxThickness = 0.005;

		// Standard Z: if rayDepth > sceneDepth + bias, ray is behind the scene surface
		if (rayDepth > sceneDepth + depthBias) {
			float depthDiff = rayDepth - sceneDepth;
			if (depthDiff < maxThickness) {
				// Occluder found!
				// Smooth fade at screen edges to avoid harsh boundary cutoffs
				float edgeFade = smoothstep(0.0, 0.05, uv.x) * smoothstep(1.0, 0.95, uv.x) *
				                 smoothstep(0.0, 0.05, uv.y) * smoothstep(1.0, 0.95, uv.y);

				// Distance fade
				float distFade = 1.0 - smoothstep(maxRayDist * 0.7, maxRayDist, currentDist);

				shadowVisibility = mix(1.0, 0.0, edgeFade * distFade);
				break;
			}
		}

		// Advance ray step
		currentDist += stepDist;
	}

	return shadowVisibility;
}

#endif // BRASSICA_SCREEN_SPACE_SHADOWS_GLSL
