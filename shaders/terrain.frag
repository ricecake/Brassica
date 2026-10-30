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

	// OrganicMacroCell mCell = generateOrganicMacroCell(worldXZ, 0.5, 3);
	// OrganicPlateData mCell = getOrganicPlate(worldXZ*(0.5+0.5*cross_noise_fbm(worldPos*0.00005, 2, sin(uTime*0.001+0.5*dot_noise(worldPos*0.0005, 0.01*uTime))*0.1).xz), 0.00025, 3.0);


	// Progressive multi-tier distance fading for normal perturbations (tapers off faster to avoid specular shimmer/aliasing):
	float microNormalFade  = 1.0 - smoothstep(15.0, 120.0, camDist);
	float mediumNormalFade = 1.0 - smoothstep(40.0, 400.0, camDist);
	float macroNormalFade  = 1.0 - smoothstep(100.0, 1000.0, camDist);

	// Progressive multi-tier distance fading for coloration (remains visible much farther into the distance):
	float microColorFade   = 1.0 - smoothstep(25.0, 350.0, camDist);
	float mediumColorFade  = 1.0 - smoothstep(80.0, 1200.0, camDist);
	float macroColorFade   = 1.0 - smoothstep(200.0, 3000.0, camDist);

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
			vec3 dummyGrad;
			float mossNoise = dot_noise(vec3(worldXZ * 1.5, 0.0), 0.0, dummyGrad);
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

		// Determine forest density and appropriate tree species canopy profile using unified tree selection:
		// 0: Pine/Evergreen, 1: Deciduous, 2: Palm/Tropical, 3: Cactus/Arid Shrub
		uint treeType = getWhittakerTreeType(biomeIdx, temp, moisture);
		float forestDensity = smoothstep(0.25, 0.75, moisture) * smoothstep(0.50, 0.88, N.y);
		vec3 leafColor = vec3(0.18, 0.48, 0.14); // Default deciduous

		if (treeType == 3u) {
			forestDensity = smoothstep(0.10, 0.35, moisture) * 0.4 * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.22, 0.45, 0.24);
		} else if (treeType == 2u) {
			forestDensity = smoothstep(0.30, 0.80, moisture) * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.15, 0.55, 0.20);
		} else if (treeType == 0u) {
			forestDensity = smoothstep(0.20, 0.65, moisture) * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.08, 0.30, 0.12);
		} else {
			forestDensity = smoothstep(0.25, 0.75, moisture) * smoothstep(0.50, 0.88, N.y);
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

			mat3 microWarpGrad;
			vec2 forestGrad = vec2(0.0);
			vec3 g1 = vec3(0.0);
			vec3 g2 = vec3(0.0);
			vec3 g3 = vec3(0.0);
			float fNoise1 = 1.0;
			float fNoise2 = 1.0;
			float fn3 = 1.0;//dot_noise(vec3(squatWarpedXZ * 2.000, 0.0), 0.0, g3);


			if (treeType == 0u) { // Pine / Evergreen: Conical peaked crowns with vertical branch texture
				forestGrad = g1.xz * (1.3 * 0.50 * macroCanopyFade) +
				             g2.xz * (0.35 * mediumCanopyFade) +
				             g3.xz * (0.15 * microCanopyFade);
			} else if (treeType == 2u) { // Tropical / Palm: Broad umbrella domes
				forestGrad = g1.xz * (0.50 * macroCanopyFade) +
				             g2.xz * (0.35 * mediumCanopyFade) +
				             g3.xz * (0.20 * microCanopyFade);
			} else if (treeType == 3u) { // Cactus / Arid Scrub: Scattered columnar bumps
				forestGrad = g2.xz * (0.60 * mediumCanopyFade) +
				             g3.xz * (0.40 * microCanopyFade);
			} else { // Deciduous: Rounded canopy clusters
				forestGrad = g1.xz * (0.45 * macroCanopyFade) +
				             g2.xz * (0.35 * mediumCanopyFade) +
				             g3.xz * (0.20 * microCanopyFade);
			}

			perturbedGradSum += forestGrad * proxyAmount;

			// Gap crevice shadowing between adjacent tree crowns
			// float crownCrevice = smoothstep(-0.4, 0.4, fNoise1 * 0.65 + fNoise2 * 0.35);
			float crownCrevice = smoothstep(-0.4, 0.4, fNoise1 * 0.65 + fNoise2 * 0.35);
			vec3 canopyColor = mix(leafColor * 0.55, leafColor * 1.15, crownCrevice);

			// Blend canopy color and increase surface roughness in dense forest areas
			finalAlbedo = mix(finalAlbedo, canopyColor, proxyAmount * 0.70);
			roughness = mix(roughness, clamp(roughness + 0.22, 0.75, 0.95), proxyAmount);
		}
	}

	// -------------------------------------------------------------------------
	// 2. Macro Tier: Broad geological shape & cliff normal structure
	// -------------------------------------------------------------------------
	if (macroColorFade > 0.001 || macroNormalFade > 0.001) {
		vec3 macroGrad3D;
		float macroNoise = 1.0;//dot_noise(vec3(worldXZ * 0.35, 0.0), 0.0, macroGrad3D);
		vec2 macroGrad = macroGrad3D.xz;

		float macroColorWeight  = macroColorFade * 0.35;
		float macroNormalWeight = macroNormalFade * 0.35;

		if (rockW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.8 + 0.35 * smoothstep(-0.5, 0.5, macroNoise)), rockW * macroColorWeight);
			perturbedGradSum += macroGrad * 0.35 * rockW * macroNormalWeight;
		}
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.9 + 0.2 * macroNoise), grassW * macroColorWeight);
		}
	}

	// -------------------------------------------------------------------------
	// 3. Medium Tier: Rock strata, sand ripples & grass ground normals
	// -------------------------------------------------------------------------
	if (mediumColorFade > 0.001 || mediumNormalFade > 0.001) {
		vec3 mediumGrad3D;
		mat3 microWarpGrad;
		// vec3 microNoiseWarp = cross_noise(worldPos, 0.0, microWarpGrad);
		vec3 warp = vec3(1.0);//2*(microWarpGrad*microNoiseWarp);

		float mediumNoise = 1.0;//dot_noise(vec3(worldXZ * 3.2, 0.0), 0.0, mediumGrad3D);
		vec2 mediumGrad = mediumGrad3D.xz;

		float mediumColorWeight  = mediumColorFade * 0.55;
		float mediumNormalWeight = mediumNormalFade * 0.55;

		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.15 * mediumNoise), grassW * mediumColorWeight);
			perturbedGradSum += mediumGrad * 0.22 * grassW * mediumNormalWeight;
		}

		if (rockW > 0.001) {
			perturbedGradSum += mediumGrad * 0.35 * rockW * mediumNormalWeight;
			float crevice = smoothstep(-0.5, 0.5, mediumNoise);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.75 + 0.25 * crevice), rockW * mediumColorWeight);
		}

		if (sandW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.08 * mediumNoise), sandW * mediumColorWeight);
			perturbedGradSum += mediumGrad * 0.15 * sandW * mediumNormalWeight;
		}
	}

	// -------------------------------------------------------------------------
	// 4. Micro Tier: Fine soil texture & bumpy close-up ground
	// -------------------------------------------------------------------------
	if (microColorFade > 0.001 || microNormalFade > 0.001) {
		vec3 microGrad3D;
		mat3 microWarpGrad;
		vec3 microNoiseWarp = vec3(0.0);//cross_noise(worldPos, 0.0, microWarpGrad);
		float microNoise = 1.0;//dot_noise(0.5*(microWarpGrad*microNoiseWarp)+vec3(worldXZ * 12.0, 0.0), 0.0, microGrad3D);
		vec2 microGrad = microGrad3D.xz;

		float microColorWeight  = microColorFade;
		float microNormalWeight = microNormalFade;

		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.1 * microNoise), grassW * microColorWeight);
			roughness = mix(roughness, clamp(roughness + 0.1 * microNoise, 0.4, 0.95), grassW * microColorWeight);
			perturbedGradSum += microGrad * 0.25 * grassW * microNormalWeight;
		}

		if (rockW > 0.001) {
			perturbedGradSum += microGrad * 0.25 * rockW * microNormalWeight;
		}

		if (sandW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * microNoise), sandW * microColorWeight);
		}

		if (snowW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * microNoise), snowW * microColorWeight);
			glint = mix(glint, clamp(glint + 0.3 * step(0.7, microNoise), 0.0, 1.0), snowW * microColorWeight);
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

	// finalAlbedo.r = mCell.regionID;
	// finalAlbedo *= smoothstep(0, 0.1, mCell.edgeDist);
	// finalAlbedo = vec3(0.5+0.5*(dot_noise_dir(worldPos*0.25, uTime, normalize(cross(vec3(sin(uTime*0.1),1,0), normalize(N))))*0.25), 0, 0);
	// finalAlbedo = vec3(0.5+0.5*(dot_noise(worldPos*0.25, uTime)*0.25), 0, 0);
	outPosition = vec4(inPosition, 0.0);
	outNormal   = vec4(normalize(N), 1.0);
	outAlbedo   = vec4(finalAlbedo, inAlbedo.a > 0.0 ? inAlbedo.a : 1.0);
	outMaterial = vec4(metallic, roughness, glint, inMaterial.a);
}
