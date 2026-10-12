#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "lygia/generative/psrdnoise.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;
layout(location = 3) in vec4 inMaterial;
layout(location = 4) in vec4 inMaterialWeights; // x=grass, y=rock, z=sand, w=snow

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

void main() {
	vec3 worldPos = inPosition + uCameraPosition.xyz;
	vec2 worldXZ = worldPos.xz;
	float camDist = length(inPosition);

	// Progressive multi-tier distance fading:
	// - Micro tier (0m to 120m): High-frequency fine texture & cracks
	// - Medium tier (0m to 400m): Medium-scale rock strata, sand ripples & grass variation
	// - Macro tier (0m to 1000m): Low-frequency geological formations & shape-providing normal perturbations
	float microFade  = 1.0 - smoothstep(15.0, 120.0, camDist);
	float mediumFade = 1.0 - smoothstep(40.0, 400.0, camDist);
	float macroFade  = 1.0 - smoothstep(100.0, 1000.0, camDist);

	vec3 N = normalize(inNormal);
	vec3 finalAlbedo = inAlbedo.rgb;
	float metallic = inMaterial.r;
	float roughness = inMaterial.g;
	float glint = inMaterial.b;

	vec4 weights = inMaterialWeights;
	float grassW = weights.x;
	float rockW  = weights.y;
	float sandW  = weights.z;
	float snowW  = weights.w;

	// Tangent space basis construction relative to geometric surface normal
	vec3 T = normalize(cross(N, abs(N.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
	vec3 B = cross(N, T);

	vec2 perturbedGradSum = vec2(0.0);

	// 1. Macro Tier (0m - 1000m): Provides broad geological shape, cliff banding, and distant normal structure
	if (macroFade > 0.001) {
		vec2 mGrad1, mGrad2;
		float mNoise1 = psrdnoise(worldXZ * 0.25, vec2(0.0), 0.0, mGrad1);
		float mNoise2 = psrdnoise(worldXZ * 0.8,  vec2(0.0), 0.0, mGrad2);
		float macroNoise = mNoise1 * 0.65 + mNoise2 * 0.35;
		vec2 macroGrad = mGrad1 * 0.65 + mGrad2 * 0.35;

		float macroWeight = macroFade * 0.35;
		if (rockW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.8 + 0.35 * smoothstep(-0.5, 0.5, macroNoise)), rockW * macroWeight);
			perturbedGradSum += macroGrad * 0.35 * rockW * macroWeight;
		}
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.9 + 0.2 * macroNoise), grassW * macroWeight);
		}
	}

	// 2. Medium Tier (0m - 400m): Medium-frequency rock strata, sand ripples, grass color variations
	if (mediumFade > 0.001) {
		float mediumWeight = mediumFade * 0.55;

		if (grassW > 0.001) {
			vec2 gGrad1;
			float gNoise1 = psrdnoise(worldXZ * 3.5, vec2(0.0), 0.0, gGrad1);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.15 * gNoise1), grassW * mediumWeight);
		}

		if (rockW > 0.001) {
			vec2 rGrad1, rGrad2;
			float rNoise1 = psrdnoise(worldXZ * 2.5, vec2(0.0), 0.0, rGrad1);
			float rNoise2 = psrdnoise(worldXZ * 8.0, vec2(0.0), 0.0, rGrad2);
			float rockNoise = rNoise1 * 0.6 + rNoise2 * 0.4;
			vec2 rockGrad = rGrad1 * 0.6 + rGrad2 * 0.4;

			perturbedGradSum += rockGrad * 0.35 * rockW * mediumWeight;
			float crevice = smoothstep(-0.5, 0.5, rockNoise);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.75 + 0.25 * crevice), rockW * mediumWeight);
		}

		if (sandW > 0.001) {
			vec2 sGrad1;
			float sNoise1 = psrdnoise(worldXZ * 6.0, vec2(0.0), 0.0, sGrad1);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.08 * sNoise1), sandW * mediumWeight);
			perturbedGradSum += sGrad1 * 0.15 * sandW * mediumWeight;
		}
	}

	// 3. Micro Tier (0m - 120m): High-frequency fine texture & cracks
	if (microFade > 0.001) {
		float microWeight = microFade;

		if (grassW > 0.001) {
			vec2 gGrad2;
			float gNoise2 = psrdnoise(worldXZ * 11.0, vec2(0.0), 0.0, gGrad2);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.1 * gNoise2), grassW * microWeight);
			roughness = mix(roughness, clamp(roughness + 0.1 * gNoise2, 0.4, 0.95), grassW * microWeight);
		}

		if (rockW > 0.001) {
			vec2 rGrad3;
			float rNoise3 = psrdnoise(worldXZ * 24.0, vec2(0.0), 0.0, rGrad3);
			perturbedGradSum += rGrad3 * 0.25 * rockW * microWeight;
		}

		if (sandW > 0.001) {
			float sNoise2 = psrdnoise(worldXZ * 20.0);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * sNoise2), sandW * microWeight);
		}

		if (snowW > 0.001) {
			float snowNoise = psrdnoise(worldXZ * 15.0);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * snowNoise), snowW * microWeight);
			glint = mix(glint, clamp(glint + 0.3 * step(0.7, snowNoise), 0.0, 1.0), snowW * microWeight);
		}
	}

	// Compute final perturbed surface normal if any perturbation was accumulated
	if (length(perturbedGradSum) > 0.0001) {
		vec3 unnormalizedN = N + (T * perturbedGradSum.x + B * perturbedGradSum.y);
		float lenN = length(unnormalizedN);
		N = unnormalizedN / max(lenN, 0.001);

		// Toksvig roughness filtering for perturbed normals:
		// Converts micro-facet normal variance into specular roughness increase
		float toksvigVariance = (1.0 - clamp(1.0 / max(lenN, 0.001), 0.0, 1.0)) / max(lenN, 0.001);
		roughness = sqrt(clamp(roughness * roughness + toksvigVariance, 0.0, 1.0));
	}

	outPosition = vec4(inPosition, 0.0);
	outNormal   = vec4(normalize(N), 1.0);
	outAlbedo   = vec4(finalAlbedo, inAlbedo.a > 0.0 ? inAlbedo.a : 1.0);
	outMaterial = vec4(metallic, roughness, glint, inMaterial.a);
}
