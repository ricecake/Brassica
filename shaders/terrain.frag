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


	vec3 N = normalize(inNormal);
	vec3 finalAlbedo = inAlbedo.rgb;
	float metallic = inMaterial.r;
	float roughness = inMaterial.g;
	float glint = inMaterial.b;

	// View vector and grazing angle for screen-space footprint:
	vec3 viewDir = -normalize(inPosition);
	float NdotV = clamp(abs(dot(N, viewDir)), 0.20, 1.0);
	float effDist = camDist / NdotV;

	// -------------------------------------------------------------------------
	// Unified 3-Tier Stationary Base Noises (Macro, Medium, Micro)
	// Evaluated at fixed world-space frequencies to eliminate texture swimming.
	// We extract both value and analytical gradient in just three invocations.
	// -------------------------------------------------------------------------
	// 1. Construct the TBN frame
	vec3 referenceUp = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 T = normalize(cross(referenceUp, N));
	vec3 B = normalize(cross(N, T));

	OrganicPlateData mCell = getOrganicPlate(worldXZ * dot(B+N, vec3(0,1,0)), 0.00025, 3.0);

	vec3 noisePos = mCell.barycentric;

	vec3 macroGrad3D;
	float macroVal = dot_noise_fbm(vec3(noisePos.xz * 0.035, noisePos.y * 0.018), 3, 0.0, macroGrad3D);

	vec3 mediumGrad3D;
	float mediumVal = dot_noise_fbm(vec3(noisePos.xz * 0.22, noisePos.y * 0.11), 2, 37.0, mediumGrad3D);

	vec3 microGrad3D;
	float microVal = dot_noise_fbm(vec3(noisePos.xz * 1.40, noisePos.y * 0.70), 2, 83.0, microGrad3D);

/*
	vec3 scaled_pos = noisePos;//vec3(noisePos.xz * 0.022, noisePos.y * 0.011);
	float footprint = length(dFdx(scaled_pos)) + length(dFdy(scaled_pos));
	vec4 macro = simplex_flow_noise_grad(vec3(noisePos.xz * 0.035, noisePos.y * 0.018), vec2(0, 1), 256);
	// float medium = simplex_fbm_curl(scaled_pos, 8, uTime, N, footprint, 512);
	// float medium = phasimpleNoise(noisePos.xz*0.01, normalize(B.xz), 0.50*uTime);
	// vec2 Z = phasorNoise3D(vec3(noisePos.xz * 0.035, noisePos.y * 0.018), 1.0, normalize(macro.yzw), B, 0.0, uTime);
	// medium = Z.x/length(Z);
	vec4 micro = simplex_flow_noise_grad(vec3(noisePos.xz * 1.40, noisePos.y * 0.70), vec2(1, 0), 64);

	vec3 macroGrad3D = macro.yzw;
	// float macroVal = macro.x;// = dot_noise_fbm(vec3(noisePos.xz * 0.035, noisePos.y * 0.018), 3, 0.0, macroGrad3D);
	float macroVal = dot_noise_fbm(vec3(noisePos.xz * 0.035, noisePos.y * 0.018), 3, 0.0, macroGrad3D);

	vec3 mediumGrad3D = macro.yzw;
	// float mediumVal = medium;//dot_noise_fbm(vec3(noisePos.xz * 0.22, noisePos.y * 0.11), 2, 37.0, mediumGrad3D);
	float mediumVal = dot_noise_fbm(vec3(noisePos.xz * 0.22, noisePos.y * 0.11), 2, 37.0, mediumGrad3D);

	vec3 microGrad3D = micro.yzw;
	// float microVal = micro.x;//dot_noise_fbm(vec3(noisePos.xz * 1.40, noisePos.y * 0.70), 2, 83.0, microGrad3D);
	float microVal = dot_noise_fbm(vec3(noisePos.xz * 1.40, noisePos.y * 0.70), 2, 83.0, microGrad3D);
*/
	// Scale analytical gradients by base frequency (chain rule: d(noise)/d(noisePos) = d(noise)/dp * freq)
	vec2 macroGrad  = macroGrad3D.xz * 0.035;
	vec2 mediumGrad = mediumGrad3D.xz * 0.22;
	vec2 microGrad  = microGrad3D.xz * 1.40;

	// -------------------------------------------------------------------------
	// 3-Band Distance Cross-Fading: One fading out, one sustaining, one fading in.
	// Near transition (~15m to ~120m): Micro fades out, Medium sustains, Macro fades in.
	// Sum of active noise weight is preserved to maintain consistent surface complexity.
	// -------------------------------------------------------------------------
	float nearBlend = smoothstep(300.0, 900.0, effDist);
	float wMicroColor  = 1.0 - nearBlend;
	float wMacroColor  = nearBlend;
	float wMediumColor = 1.0;

	// Far transition (~120m to ~700m): Medium fades out, Macro sustains.
	float farBlend = smoothstep(900.0, 1500.0, effDist);
	wMediumColor *= (1.0 - farBlend);

	// Horizon transition (>700m to 2500m): Macro tapers smoothly to base terrain albedo.
	float horizonFade = 1.0 - smoothstep(1500.0, 5000.0, effDist);
	wMacroColor *= horizonFade;

	// Normal perturbation weights (fade slightly earlier to prevent specular shimmer / subpixel aliasing):
	float nearNormBlend = smoothstep(10.0, 500.0, effDist);
	float wMicroNorm  = 1.0 - nearNormBlend;
	float wMacroNorm  = nearNormBlend;
	float wMediumNorm = 1.0;

	float farNormBlend = smoothstep(50.0, 800.0, effDist);
	wMediumNorm *= (1.0 - farNormBlend);

	float horizonNormFade = 1.0 - smoothstep(250.0, 1500.0, effDist);
	wMacroNorm *= horizonNormFade;

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

	WhittakerBiome wb;
	vec3 palA, palB, palC, palD;
	if (hasWeather) {
		float precipType = (temp < 0.25) ? 1.0 : ((temp < 0.35) ? 0.5 : 0.0);
		wb = evaluateWhittakerBiome(temp, moisture, severity, precipType);
		palA = wb.palA;
		palB = wb.palB;
		palC = wb.palC;
		palD = wb.palD;
	} else {
		uint bIdx = params.biomeIndex;
		wb.biomeIndex = float(bIdx);
		wb.roughness = 0.75;
		wb.snowCover = 0.0;
		getWhittakerBiomePalette(bIdx, palA, palB, palC, palD);
		wb.palA = palA;
		wb.palB = palB;
		wb.palC = palC;
		wb.palD = palD;
		wb.color = evalCosinePalette(palA, palB, palC, palD, 0.65);
	}

	// Evaluate the cosine palette with parameter t modulated by moisture, slope, and procedural noise.
	// As distance increases, modulation seamlessly transitions from medium-scale variegation to broad macro-scale geographic variation.
	float palNoise = mix(mediumVal, macroVal, nearBlend);
	float palT = clamp(0.60 + 0.25 * palNoise + 0.15 * (moisture - 0.5) + 0.10 * (N.y - 0.7), 0.0, 1.0);
	vec3 paletteColor = evalCosinePalette(palA, palB, palC, palD, palT);

	// 1. Climate biome color blending: enhance greenness in moist regions
	if (grassW > 0.01) {
		vec3 lushGreen = mix(paletteColor, vec3(0.12, 0.48, 0.10), smoothstep(0.35, 0.75, moisture) * 0.5);
		finalAlbedo = mix(finalAlbedo, lushGreen, grassW * 0.70);
		roughness = mix(roughness, clamp(wb.roughness + 0.08 * (mediumVal * wMediumColor + microVal * wMicroColor), 0.35, 0.95), grassW * 0.5);
		perturbedGradSum += (macroGrad * wMacroNorm + mediumGrad * wMediumNorm) * (0.12 * grassW);
	}

	// 2. Patches of green moss / damp vegetation on wet cliff ledges and crevices
	if (rockW > 0.01 && moisture > 0.20 && (wMediumColor > 0.001 || wMacroColor > 0.001)) {
		float ledgeFactor = smoothstep(0.25, 0.75, N.y);
		float mossNoise = mediumVal;
		float mossAmount = rockW * ledgeFactor * smoothstep(0.25, 0.65, moisture) * smoothstep(-0.3, 0.4, mossNoise) * (wMediumColor * 0.7 + wMacroColor * 0.3);
		vec3 mossBase = evalCosinePalette(palA, palB, palC, palD, 0.35 + 0.20 * mediumVal);
		vec3 mossColor = mix(mossBase, vec3(0.10, 0.52, 0.12), moisture * 0.7);
		finalAlbedo = mix(finalAlbedo, mossColor, mossAmount * 0.75);
		roughness = mix(roughness, 0.85, mossAmount * 0.5);
		perturbedGradSum += mediumGrad * (0.15 * mossAmount * wMediumNorm);
	}

	// -------------------------------------------------------------------------
	// 1. Procedural Tree Forest Canopy Proxy & Feature-Size Normal Perturbation
	// -------------------------------------------------------------------------
	// Serves as a 3D canopy standin at distance where explicit foliage meshes
	// fade out, providing normal mapping, crevice shadowing, and roughness tweaks.
	if (hasWeather && grassW > 0.02 && N.y > 0.50 && worldPos.y > params.waterLevel - 0.5) {
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

			vec2 forestGrad = vec2(0.0);
			if (treeType == 0u) { // Pine / Evergreen: Conical peaked crowns with vertical branch texture
				forestGrad = macroGrad * (1.3 * 0.50 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.15 * wMicroNorm);
			} else if (treeType == 2u) { // Tropical / Palm: Broad umbrella domes
				forestGrad = macroGrad * (0.50 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.20 * wMicroNorm);
			} else if (treeType == 3u) { // Cactus / Arid Scrub: Scattered columnar bumps
				forestGrad = mediumGrad * (0.60 * wMediumNorm) +
				             microGrad * (0.40 * wMicroNorm);
			} else { // Deciduous: Rounded canopy clusters
				forestGrad = macroGrad * (0.45 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.20 * wMicroNorm);
			}

			perturbedGradSum += forestGrad * proxyAmount;

			// Blend leafColor with the cosine palette canopy highlight
			vec3 canopyHighlight = evalCosinePalette(palA, palB, palC, palD, 0.80 + 0.20 * mediumVal);
			leafColor = mix(leafColor, canopyHighlight, 0.40);

			// Gap crevice shadowing between adjacent tree crowns
			float crownCrevice = smoothstep(-0.4, 0.4, macroVal * 0.65 + mediumVal * 0.35);
			vec3 canopyColor = mix(leafColor * 0.55, leafColor * 1.15, crownCrevice);

			// Blend canopy color and increase surface roughness in dense forest areas
			finalAlbedo = mix(finalAlbedo, canopyColor, proxyAmount * 0.70);
			roughness = mix(roughness, clamp(roughness + 0.22, 0.75, 0.95), proxyAmount);
		}
	}

	// -------------------------------------------------------------------------
	// 2. Macro Tier: Broad geological shape & cliff normal structure
	// -------------------------------------------------------------------------
	if (wMacroColor > 0.001 || wMacroNorm > 0.001) {
		if (rockW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.8 + 0.35 * smoothstep(-0.5, 0.5, macroVal)), rockW * wMacroColor * 0.35);
			perturbedGradSum += macroGrad * 0.35 * rockW * wMacroNorm;
		}
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.9 + 0.2 * macroVal), grassW * wMacroColor * 0.35);
		}
	}

	// -------------------------------------------------------------------------
	// 3. Medium Tier: Rock strata, sand ripples & grass ground normals
	// -------------------------------------------------------------------------
	if (wMediumColor > 0.001 || wMediumNorm > 0.001) {
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.15 * mediumVal), grassW * wMediumColor * 0.45);
			perturbedGradSum += mediumGrad * 0.22 * grassW * wMediumNorm;
		}

		if (rockW > 0.001) {
			perturbedGradSum += mediumGrad * 0.35 * rockW * wMediumNorm;
			float crevice = smoothstep(-0.5, 0.5, mediumVal);
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.75 + 0.25 * crevice), rockW * wMediumColor * 0.45);
		}

		if (sandW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.08 * mediumVal), sandW * wMediumColor * 0.45);
			perturbedGradSum += mediumGrad * 0.15 * sandW * wMediumNorm;
		}
	}

	// -------------------------------------------------------------------------
	// 4. Micro Tier: Fine soil texture & bumpy close-up ground
	// -------------------------------------------------------------------------
	if (wMicroColor > 0.001 || wMicroNorm > 0.001) {
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.1 * microVal), grassW * wMicroColor);
			roughness = mix(roughness, clamp(roughness + 0.1 * microVal, 0.4, 0.95), grassW * wMicroColor);
			perturbedGradSum += microGrad * 0.25 * grassW * wMicroNorm;
		}

		if (rockW > 0.001) {
			perturbedGradSum += microGrad * 0.25 * rockW * wMicroNorm;
		}

		if (sandW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * microVal), sandW * wMicroColor);
		}

		if (snowW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (1.0 + 0.05 * microVal), snowW * wMicroColor);
			glint = mix(glint, clamp(glint + 0.3 * step(0.7, microVal), 0.0, 1.0), snowW * wMicroColor);
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
	// finalAlbedo = worldPos;
	// finalAlbedo = mCell.barycentric;
	outPosition = vec4(inPosition, 0.0);
	outNormal   = vec4(normalize(N), 1.0);
	outAlbedo   = vec4(finalAlbedo, inAlbedo.a > 0.0 ? inAlbedo.a : 1.0);
	outMaterial = vec4(metallic, roughness, glint, inMaterial.a);
}
