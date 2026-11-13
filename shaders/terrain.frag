#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "terrain.glsl"
#include "helpers/wind.glsl"
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

	mat3 warpNoiseGrad;
	vec3 warpNoise = 1.0*1*cross_noise_fbm(worldPos*0.00125, 3, 12, warpNoiseGrad);
	vec3 warpCurl = get_curl(warpNoiseGrad);
	// OrganicPlateData mCellS = getOrganicPlate(((worldPos+warpNoise)).xz, 0.00625, 5.0);
	// OrganicPlateData mCellM = getOrganicPlate(((worldPos+warpNoise)).xz, 0.003125, 7.0);
	// OrganicPlateData mCellL = getOrganicPlate(((worldPos+warpNoise)).xz, 0.0015625, 11.0);

	// vec3 noisePos = clamp(mCell.barycentric, 0.0, 1.0);
	// vec3 noisePosL = vec3(mCellL.nearestCentroid.x, 0.0, mCellL.nearestCentroid.y) + ((2.0*mCellL.barycentric)-1.0);
	// vec3 noisePosM = vec3(mCellM.nearestCentroid.x, 0.0, mCellM.nearestCentroid.y) + ((2.0*mCellM.barycentric)-1.0);
	// vec3 noisePosS = vec3(mCellS.nearestCentroid.x, 0.0, mCellS.nearestCentroid.y) + ((2.0*mCellS.barycentric)-1.0);
	// PUT THIS IN THE debris shader for tree cannopy

	// vec3 noisePosL = vec3(mCellL.nearestCentroid.x, 0.0, mCellL.nearestCentroid.y) + (mCellL.barycentric-vec3(1.0/3.0));
	// vec3 noisePosM = vec3(mCellM.nearestCentroid.x, 0.0, mCellM.nearestCentroid.y) + (mCellM.barycentric-vec3(1.0/3.0));
	// vec3 noisePosS = vec3(mCellS.nearestCentroid.x, 0.0, mCellS.nearestCentroid.y) + (mCellS.barycentric-vec3(1.0/3.0));

	// 2. Wind Alignment & Terrain Exposure to Wind
	// Surface prevailing wind vector from latitude:
	float lat = clamp(worldPos.z / (FAKE_PLANET_RADIUS * 1.5707963), -1.0, 1.0);
	vec2 pWind = getPrevailingWind(lat);
	vec3 windDir3D = vec3(pWind.x, 0.0, pWind.y);
	float windLen = length(windDir3D);
	if (windLen > 1e-4) {
		windDir3D /= windLen;
	} else {
		windDir3D = vec3(1.0, 0.0, 0.0);
	}

	// Tangent plane wind vector projection for wave alignment
	vec3 windTangent = normalize(windDir3D - N * dot(windDir3D, N));
	if (length(windTangent) < 1e-4) {
		windTangent = T;
	}

	// Surface exposure to wind:
	// Wind blowing against the slope: N dot (-windDir3D) > 0 is the exposed windward face.
	// Leeward / back sides of hills face away from incoming wind (N dot (-windDir3D) < 0).
	// Exposure factor slows phase speed down to 0.25x on the back side of hills.
	float slopeExposure = dot(N, -windDir3D);
	float windExposure = mix(0.25, 1.35, smoothstep(-0.6, 0.6, slopeExposure));

	// Phase time scales with surface wind exposure:
	float exposedTime = uTime * windExposure;

// -------------------------------------------------------------------------
	// Unified 3-Tier Dynamic LOD Base Noises
	// -------------------------------------------------------------------------

	// Configure the dynamic LOD. You can tweak lodDistanceScale to control how fast layers shift.
	float lodDistanceScale = 0.05;
	float baseFrequency = 1.0;
	float growthFactor = 2.0; // Octaves double in size

	NoiseCascade cascade = calculateDynamicLOD(
		effDist * lodDistanceScale,
		baseFrequency,
		growthFactor
		// 0.25, 0.50, 0.25 // Fade in, Hold, Fade out
	);

	// Apply dynamic scales to the noise coordinates
	vec3 noisePosL = warpNoise + worldPos * cascade.frequencies.x; // Macro
	vec3 noisePosM = warpNoise + worldPos * cascade.frequencies.y; // Medium
	vec3 noisePosS = warpNoise + worldPos * cascade.frequencies.z; // Micro

	// Scale wave speeds by frequency to prevent distance waves from looking hyper-fast
	// Can we make the windspeed biome dependent?
	float timeL = exposedTime * cascade.frequencies.x * 0.0625;
	float timeM = exposedTime * cascade.frequencies.y * 0.0625;
	float timeS = exposedTime * cascade.frequencies.z * 0.0625;

	// Align phasor spatial frequency F and advection velocity W with windTangent
	vec3 windWaveF_L = normalize(windTangent + warpCurl);// * 0.2);
	vec3 windWaveF_M = normalize(windTangent + warpCurl);// * 0.4);
	vec3 windWaveF_S = normalize(windTangent + warpCurl);// * 0.6);

	vec3 macroGrad3D;
	vec2 macroValPhase = phasorNoise3DGrad(noisePosL, 1.0, windWaveF_L, windTangent, 0.68, timeL, macroGrad3D);
	float macroVal = macroValPhase.x / max(length(macroValPhase), 0.0001);

	vec3 mediumGrad3D;
	vec2 mediumValPhase = phasorNoise3DGrad(noisePosM, 1.0, windWaveF_M, windTangent, 0.68, timeM, mediumGrad3D);
	float mediumVal = mediumValPhase.x / max(length(mediumValPhase), 0.0001);

	vec3 microGrad3D;
	vec2 microValPhase = phasorNoise3DGrad(noisePosS, 1.0, windWaveF_S, windTangent, 0.68, timeS, microGrad3D);
	float microVal = microValPhase.x / max(length(microValPhase), 0.0001);

	// CHAIN RULE FIX: Multiply the returned gradient by the layer's frequency scale.
	// Without this, your normals will randomly flatten or exaggerate as the scale shifts.
	vec2 macroGrad  = macroGrad3D.xz * cascade.frequencies.x;
	vec2 mediumGrad = mediumGrad3D.xz * cascade.frequencies.y;
	vec2 microGrad  = microGrad3D.xz * cascade.frequencies.z;

	macroGrad  /= (1.0 + length(macroGrad));
	mediumGrad /= (1.0 + length(mediumGrad));
	microGrad  /= (1.0 + length(microGrad));

	// Map cascade weights directly to your color and normal blend variables
	float wMacroColor  = cascade.weights.x;
	float wMediumColor = cascade.weights.y;
	float wMicroColor  = cascade.weights.z;

	float wMacroNorm  = cascade.weights.x;
	float wMediumNorm = cascade.weights.y;
	float wMicroNorm  = cascade.weights.z;

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
// Evaluate the cosine palette with parameter t modulated by moisture, slope, and procedural noise.
	// This linearly blends all 3 dynamic layers to determine the biome palette offset.
	float palNoise = (macroVal * wMacroColor) + (mediumVal * wMediumColor) + (microVal * wMicroColor);
	float palT = clamp(0.60 + 0.25 * palNoise + 0.15 * (moisture - 0.5) + 0.10 * (N.y - 0.7), 0.0, 1.0);
	vec3 paletteColor = evalCosinePalette(palA, palB, palC, palD, palT);

	// 1. Climate biome color blending: enhance greenness in moist regions
	if (grassW > 0.01) {
		vec3 lushGreen = mix(paletteColor, vec3(0.12, 0.48, 0.10), smoothstep(0.35, 0.75, moisture) * 0.5);

		// Wind gust response on grass:
		// When grass bends under wind gusts, blades reflect light differently, exposing lighter
		// undersides and shifting roughness (specular sheen) dynamically along the wave phase.
		float grassWindWave = microVal * 0.6 + mediumVal * 0.4;
		// Lightness brightening (+15% to -10%) as grass leans into/away from wind
		float grassLightness = 1.0 + 0.18 * grassWindWave * windExposure;
		// Roughness reduction on bent grass creates moving specular highlights across fields
		float grassRoughnessMod = -0.15 * max(0.0, grassWindWave) * windExposure;

		lushGreen *= grassLightness;

		finalAlbedo = mix(finalAlbedo, lushGreen, grassW * 0.70);
		roughness = mix(roughness, clamp(wb.roughness + grassRoughnessMod + 0.08 * (mediumVal * wMediumColor + microVal * wMicroColor), 0.25, 0.95), grassW * 0.5);
		perturbedGradSum += (macroGrad * wMacroNorm + mediumGrad * wMediumNorm + microGrad * wMicroNorm) * (0.18 * grassW);
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
			float proxyDistFade = mix(0.35, 1.0, smoothstep(100.0, 1800.0, camDist));
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
	// finalAlbedo = mCellS.barycentric;
	outPosition = vec4(inPosition, 0.0);
	outNormal   = vec4(normalize(N), 1.0);
	outAlbedo   = vec4(finalAlbedo, inAlbedo.a > 0.0 ? inAlbedo.a : 1.0);
	outMaterial = vec4(metallic, roughness, glint, inMaterial.a);
}
