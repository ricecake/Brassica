#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "terrain.glsl"
#include "lygia/generative/psrdnoise.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;
layout(location = 3) in vec4 inMaterial;
layout(location = 4) in vec4 inMaterialWeights; // x=grass, y=rock, z=sand, w=snow

layout(push_constant) uniform TerrainPushConstants {
	uvec4 gridParams; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
	uint  clipmapIndex;
	uint  minMaxIndex;
	uint  biomeIndex;
	float waterLevel;
	uint  weatherBiomeIndex;
} params;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

void main() {
	vec3 worldPos = inPosition + uCameraPosition.xyz;
	vec2 worldXZ = worldPos.xz;
	float camDist = length(inPosition);

	// Progressive multi-tier distance fading for general terrain texturing:
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

	vec2 perturbedGradSum = vec2(0.0);

	// Sample climate weather biome parameters if available
	float cloudCoverage = 0.0;
	float temp = 0.5;
	float moisture = 0.5;
	float severity = 0.0;
	bool hasWeather = false;

	if (params.weatherBiomeIndex > 0u) {
		vec4 weatherSample = sampleTerrainWeatherBiome(params.weatherBiomeIndex, worldPos);
		cloudCoverage = weatherSample.r;
		temp = weatherSample.g;
		moisture = weatherSample.b;
		severity = weatherSample.a;
		hasWeather = true;
	}

	if (hasWeather) {
		float precipType = (temp < 0.25) ? 1.0 : ((temp < 0.35) ? 0.5 : 0.0);
		WhittakerBiome wb = evaluateWhittakerBiome(temp, moisture, severity, precipType);

		// 1. Climate biome color blending: enhance greenness in moist regions
		if (grassW > 0.01) {
			vec3 lushGreen = mix(wb.color, vec3(0.12, 0.48, 0.10), smoothstep(0.35, 0.75, moisture));
			finalAlbedo = mix(finalAlbedo, lushGreen, grassW * 0.65);
			roughness = mix(roughness, wb.roughness, grassW * 0.5);
		}

		// 2. Patches of green moss / damp vegetation on wet cliff ledges and crevices
		if (rockW > 0.01 && moisture > 0.20) {
			float ledgeFactor = smoothstep(0.25, 0.75, N.y);
			float mossNoise = psrdnoise(worldXZ * 1.5);
			float mossAmount = rockW * ledgeFactor * smoothstep(0.25, 0.65, moisture) * smoothstep(-0.3, 0.4, mossNoise);
			vec3 mossColor = mix(vec3(0.18, 0.40, 0.14), vec3(0.10, 0.52, 0.12), moisture);
			finalAlbedo = mix(finalAlbedo, mossColor, mossAmount * 0.75);
			roughness = mix(roughness, 0.85, mossAmount * 0.5);
		}
	}

	// -------------------------------------------------------------------------
	// 1. Procedural Tree Forest Canopy Proxy & Feature-Size Normal Perturbation
	// -------------------------------------------------------------------------
	// Serves as a 3D canopy standin at distance where explicit foliage meshes
	// fade out, providing normal mapping, crevice shadowing, and roughness tweaks.
	if (hasWeather && grassW > 0.02 && N.y > 0.50 && worldPos.y > params.waterLevel - 0.5) {
		float precipType = (temp < 0.25) ? 1.0 : ((temp < 0.35) ? 0.5 : 0.0);
		WhittakerBiome wb = evaluateWhittakerBiome(temp, moisture, severity, precipType);
		uint biomeIdx = uint(clamp(wb.biomeIndex, 0.0, 9.0));

		// Determine forest density and appropriate tree species canopy profile:
		// 0: Pine/Evergreen (Taiga/Tundra/Rainforest)
		// 1: Deciduous (Temperate Deciduous/Woodland)
		// 2: Palm/Tropical (Tropical Rainforest/Savanna)
		// 3: Cactus/Arid Shrub (Subtropical Desert)
		uint treeType = 1u;
		float forestDensity = smoothstep(0.20, 0.70, moisture) * smoothstep(0.50, 0.88, N.y);
		vec3 leafColor = vec3(0.18, 0.48, 0.14); // Default deciduous

		if (biomeIdx == 7u || moisture < 0.25) {
			treeType = 3u; // Cactus / Arid Scrub
			forestDensity = smoothstep(0.10, 0.35, moisture) * 0.4;
			leafColor = vec3(0.22, 0.45, 0.24);
		} else if (biomeIdx == 8u || biomeIdx == 9u || (temp > 0.70 && moisture > 0.60)) {
			treeType = 2u; // Palm / Tropical Canopy
			forestDensity = smoothstep(0.30, 0.80, moisture);
			leafColor = vec3(0.15, 0.55, 0.20);
		} else if (biomeIdx == 0u || biomeIdx == 1u || biomeIdx == 2u || temp < 0.38) {
			treeType = 0u; // Pine / Evergreen
			forestDensity = smoothstep(0.20, 0.65, moisture);
			leafColor = vec3(0.08, 0.30, 0.12);
		} else {
			treeType = 1u; // Deciduous
			forestDensity = smoothstep(0.25, 0.75, moisture);
			leafColor = vec3(0.18, 0.48, 0.14);
		}

		if (forestDensity > 0.01) {
			// Proxy weight increases with distance as explicit foliage mesh instances fade out beyond ~40-180m
			float proxyDistFade = mix(0.35, 1.0, smoothstep(30.0, 180.0, camDist));
			float proxyAmount = grassW * forestDensity * proxyDistFade;

			// Anti-shimmer distance fading for noise octaves:
			// Higher-frequency feature octaves fade out smoothly before reaching subpixel sampling
			// frequencies to eliminate shimmering and temporal aliasing artifacts at long range.
			float microCanopyFade  = 1.0 - smoothstep(30.0, 100.0, camDist);   // Fine leaf/needle tufts (~0.5m - 0.8m)
			float mediumCanopyFade = 1.0 - smoothstep(80.0, 300.0, camDist);   // Branch clusters (~2.0m - 3.5m)
			float macroCanopyFade  = 1.0 - smoothstep(200.0, 1200.0, camDist); // Individual tree crowns (~6.0m - 10.0m)

			vec2 forestGrad = vec2(0.0);
			float fNoise1 = 0.0;
			float fNoise2 = 0.0;

			if (treeType == 0u) { // Pine / Evergreen: Conical peaked crowns with vertical branch texture
				// Frequencies tied to physical sizes: crown ~6m (0.166), branch ~2m (0.50), needle tuft ~0.5m (2.0)
				vec2 g1, g2, g3;
				fNoise1 = psrdnoise(worldXZ * 0.166, vec2(0.0), 0.0, g1);
				fNoise2 = psrdnoise(worldXZ * 0.500, vec2(0.0), 0.0, g2);
				float fn3 = psrdnoise(worldXZ * 2.000, vec2(0.0), 0.0, g3);

				forestGrad = g1 * (1.3 * 0.50 * macroCanopyFade) +
				             g2 * (0.35 * mediumCanopyFade) +
				             g3 * (0.15 * microCanopyFade);
			} else if (treeType == 2u) { // Tropical / Palm: Broad umbrella domes
				// Frequencies tied to physical sizes: dome ~10m (0.10), fronds ~3m (0.33), frond blade ~0.8m (1.25)
				vec2 g1, g2, g3;
				fNoise1 = psrdnoise(worldXZ * 0.100, vec2(0.0), 0.0, g1);
				fNoise2 = psrdnoise(worldXZ * 0.333, vec2(0.0), 0.0, g2);
				float fn3 = psrdnoise(worldXZ * 1.250, vec2(0.0), 0.0, g3);

				forestGrad = g1 * (0.50 * macroCanopyFade) +
				             g2 * (0.35 * mediumCanopyFade) +
				             g3 * (0.20 * microCanopyFade);
			} else if (treeType == 3u) { // Cactus / Arid Scrub: Scattered columnar bumps
				// Frequencies tied to physical sizes: shrub ~3m (0.33), column ~1m (1.0)
				vec2 g2, g3;
				fNoise2 = psrdnoise(worldXZ * 0.333, vec2(0.0), 0.0, g2);
				float fn3 = psrdnoise(worldXZ * 1.000, vec2(0.0), 0.0, g3);

				forestGrad = g2 * (0.60 * mediumCanopyFade) +
				             g3 * (0.40 * microCanopyFade);
			} else { // Deciduous: Rounded canopy clusters
				// Frequencies tied to physical sizes: crown ~8m (0.125), cluster ~2.5m (0.40), leaf tuft ~0.6m (1.66)
				vec2 g1, g2, g3;
				fNoise1 = psrdnoise(worldXZ * 0.125, vec2(0.0), 0.0, g1);
				fNoise2 = psrdnoise(worldXZ * 0.400, vec2(0.0), 0.0, g2);
				float fn3 = psrdnoise(worldXZ * 1.666, vec2(0.0), 0.0, g3);

				forestGrad = g1 * (0.45 * macroCanopyFade) +
				             g2 * (0.35 * mediumCanopyFade) +
				             g3 * (0.20 * microCanopyFade);
			}

			perturbedGradSum += forestGrad * proxyAmount;

			// Gap crevice shadowing between adjacent tree crowns
			float crownCrevice = smoothstep(-0.4, 0.4, fNoise1 * 0.65 + fNoise2 * 0.35);
			vec3 canopyColor = mix(leafColor * 0.55, leafColor * 1.15, crownCrevice);

			// Blend canopy color and increase surface roughness in dense forest areas
			finalAlbedo = mix(finalAlbedo, canopyColor, proxyAmount * 0.70);
			roughness = mix(roughness, clamp(roughness + 0.22, 0.75, 0.95), proxyAmount);
		}
	}

	// -------------------------------------------------------------------------
	// 2. Macro Tier (0m - 1000m): Broad geological shape & cliff normal structure
	// -------------------------------------------------------------------------
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

	// -------------------------------------------------------------------------
	// 3. Medium Tier (0m - 400m): Rock strata, sand ripples & grass ground normals
	// -------------------------------------------------------------------------
	if (mediumFade > 0.001) {
		float mediumWeight = mediumFade * 0.55;

		if (grassW > 0.001) {
			vec2 gGrad1;
			float gNoise1 = psrdnoise(worldXZ * 3.5, vec2(0.0), 0.0, gGrad1);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.15 * gNoise1), grassW * mediumWeight);
			perturbedGradSum += gGrad1 * 0.22 * grassW * mediumWeight;
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

	// -------------------------------------------------------------------------
	// 4. Micro Tier (0m - 120m): Fine soil texture & bumpy close-up ground
	// -------------------------------------------------------------------------
	if (microFade > 0.001) {
		float microWeight = microFade;

		if (grassW > 0.001) {
			vec2 gGrad2;
			float gNoise2 = psrdnoise(worldXZ * 11.0, vec2(0.0), 0.0, gGrad2);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.1 * gNoise2), grassW * microWeight);
			roughness = mix(roughness, clamp(roughness + 0.1 * gNoise2, 0.4, 0.95), grassW * microWeight);
			perturbedGradSum += gGrad2 * 0.25 * grassW * microWeight;
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

	// -------------------------------------------------------------------------
	// 5. Dynamic Rain Wetness, Darkening, Glossiness & Glint
	// -------------------------------------------------------------------------
	if (hasWeather) {
		float rainFactor = clamp(cloudCoverage * 1.2 * smoothstep(0.15, 0.55, moisture) + severity * 0.5, 0.0, 1.0);
		if (rainFactor > 0.01) {
			// Porous wet surfaces darken when rained on
			float wetDarkening = mix(1.0, 0.65, rainFactor * (1.0 - snowW));
			finalAlbedo *= wetDarkening;

			// Wet surfaces become glossy and glisten in the rain
			roughness = mix(roughness, 0.08, rainFactor * 0.80);
			glint = mix(glint, 0.95, rainFactor * 0.85);
		}
	}

	// -------------------------------------------------------------------------
	// 6. Toksvig Roughness Filtering & Normal Perturbation Derivation
	// -------------------------------------------------------------------------
	if (length(perturbedGradSum) > 0.0001) {
		vec3 grad3D = vec3(perturbedGradSum.x, 0.0, perturbedGradSum.y);

		// Subtract the gradient and project out the component parallel to the original normal
		vec3 unnormalizedN = N - (grad3D - N * dot(grad3D, N));

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
