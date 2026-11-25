#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

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
	uint  hydroMapIndex;
} params;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;


const ivec2 HYDRO_D8_OFFSETS[8] = ivec2[8](
	ivec2( 1,  0), ivec2( 1,  1), ivec2( 0,  1), ivec2(-1,  1),
	ivec2(-1,  0), ivec2(-1, -1), ivec2( 0, -1), ivec2( 1, -1)
);

const vec2 D8_DIR_XZ[8] = vec2[8](
	vec2( 1.0,  0.0), normalize(vec2( 1.0,  1.0)), vec2( 0.0,  1.0), normalize(vec2(-1.0,  1.0)),
	vec2(-1.0,  0.0), normalize(vec2(-1.0, -1.0)), vec2( 0.0, -1.0), normalize(vec2( 1.0, -1.0))
);

vec2 decodeFlowDir(float encodedDir) {
	int dir = int(round(encodedDir));
	if (dir < 0 || dir > 7) return vec2(0.0);
	return normalize(vec2(HYDRO_D8_OFFSETS[dir]));
}

vec4 sampleHydrologySmooth(uint texIdx, vec2 worldXZ, out vec2 outFlowVec) {
	float worldExtent = 2.0 * FAKE_PLANET_RADIUS * PI;
	vec2 uv = fract((worldXZ / worldExtent) + 0.5); // Toroidal wrap matching climate_ca.comp

	ivec2 texDim = textureSize(uTextures2D[nonuniformEXT(texIdx)], 0);
	vec2 texelCoord = uv * vec2(texDim) - 0.5;
	ivec2 p00 = ivec2(floor(texelCoord));
	vec2 f = fract(texelCoord);

	ivec2 c00 = (p00 + ivec2(0, 0) + texDim) % texDim;
	ivec2 c10 = (p00 + ivec2(1, 0) + texDim) % texDim;
	ivec2 c01 = (p00 + ivec2(0, 1) + texDim) % texDim;
	ivec2 c11 = (p00 + ivec2(1, 1) + texDim) % texDim;

	// vec4 s00 = texelFetch(uTextures2D[nonuniformEXT(texIdx)], c00, 0);
	// vec4 s10 = texelFetch(uTextures2D[nonuniformEXT(texIdx)], c10, 0);
	// vec4 s01 = texelFetch(uTextures2D[nonuniformEXT(texIdx)], c01, 0);
	// vec4 s11 = texelFetch(uTextures2D[nonuniformEXT(texIdx)], c11, 0);
	vec4 s00 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], c00);
	vec4 s10 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], c10);
	vec4 s01 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], c01);
	vec4 s11 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], c11);

	// Bilinear blend for continuous channels (filledElevation, discharge, bakedDischarge)
	vec4 blended = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);

	// Decode discrete D8 indices to 2D vectors BEFORE bilinear interpolation
	int d00 = int(round(s00.b));
	int d10 = int(round(s10.b));
	int d01 = int(round(s01.b));
	int d11 = int(round(s11.b));

	vec2 v00 = (d00 >= 0 && d00 < 8) ? D8_DIR_XZ[d00] : vec2(0.0);
	vec2 v10 = (d10 >= 0 && d10 < 8) ? D8_DIR_XZ[d10] : vec2(0.0);
	vec2 v01 = (d01 >= 0 && d01 < 8) ? D8_DIR_XZ[d01] : vec2(0.0);
	vec2 v11 = (d11 >= 0 && d11 < 8) ? D8_DIR_XZ[d11] : vec2(0.0);

	outFlowVec = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
	return blended;
}


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

	vec4 weights = inMaterialWeights;
	float grassW = weights.x;
	float rockW  = weights.y;
	float sandW  = weights.z;
	float snowW  = weights.w;

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


// View vector and grazing angle for screen-space footprint:
	vec3 viewDir = -normalize(inPosition);
	float NdotV = clamp(abs(dot(N, viewDir)), 0.20, 1.0);
	float effDist = camDist / NdotV;

	// 1. Construct the TBN frame & shared stationary wave orientation
	vec3 referenceUp = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 T = normalize(cross(referenceUp, N));

	mat3 warpNoiseGrad;
	vec3 warpNoise = cross_noise_fbm(worldPos * 0.00125, 3, 12.0, warpNoiseGrad);
	vec3 warpCurl = get_curl(warpNoiseGrad);

	float lat = clamp(worldPos.z / (FAKE_PLANET_RADIUS * 1.5707963), -1.0, 1.0);
	vec2 pWind = getPrevailingWind(lat);
	vec3 windDir3D = vec3(pWind.x, 0.0, pWind.y);
	float windLen = length(windDir3D);
	windDir3D = (windLen > 1e-4) ? (windDir3D / windLen) : vec3(1.0, 0.0, 0.0);
	// windDir3D = getLocalWind(worldPos, uTimeMs);


	vec3 windTangent = windDir3D - N * dot(windDir3D, N);
	windTangent = (length(windTangent) > 1e-4) ? normalize(windTangent) : T;

	vec3 waveF = normalize(windTangent + warpCurl);
	vec3 warpedPos = warpNoise + worldPos;

	// -------------------------------------------------------------------------
	// Stationary 3-Tier Dynamic LOD Base Noises (aligned with generation.glsl)
	// -------------------------------------------------------------------------
	float lodDistanceScale = 0.001;
	float baseFrequency = 0.050;
	float growthFactor = 2.0;

	NoiseCascade cascade = calculateDynamicLOD(
		effDist * lodDistanceScale,
		baseFrequency,
		growthFactor
	);

	vec3 noisePosL = warpedPos * cascade.frequencies.x; // Macro (20m close up)
	vec3 noisePosM = warpedPos * cascade.frequencies.y; // Medium (10m close up)
	vec3 noisePosS = warpedPos * cascade.frequencies.z; // Micro (5m close up)

	vec3 macroGrad3D;
	vec2 macroValPhase = phasorNoise3DGrad(noisePosL, 1.0, waveF, windTangent, 0.68, 0.0, macroGrad3D);
	float macroVal = macroValPhase.x / max(length(macroValPhase), 0.0001);

	vec3 mediumGrad3D;
	vec2 mediumValPhase = phasorNoise3DGrad(noisePosM, 1.0, waveF, windTangent, 0.68, 0.0, mediumGrad3D);
	float mediumVal = mediumValPhase.x / max(length(mediumValPhase), 0.0001);

	vec3 microGrad3D;
	vec2 microValPhase = phasorNoise3DGrad(noisePosS, 1.0, waveF, windTangent, 0.68, 0.0, microGrad3D);
	float microVal = microValPhase.x / max(length(microValPhase), 0.0001);

	vec2 macroGrad  = macroGrad3D.xz * cascade.frequencies.x;
	vec2 mediumGrad = mediumGrad3D.xz * cascade.frequencies.y;
	vec2 microGrad  = microGrad3D.xz * cascade.frequencies.z;

	macroGrad  /= (1.0 + length(macroGrad));
	mediumGrad /= (1.0 + length(mediumGrad));
	microGrad  /= (1.0 + length(microGrad));

	float wMacroColor  = cascade.weights.x;
	float wMediumColor = cascade.weights.y;
	float wMicroColor  = cascade.weights.z;

	float wMacroNorm  = cascade.weights.x;
	float wMediumNorm = cascade.weights.y;
	float wMicroNorm  = cascade.weights.z;

	vec2 perturbedGradSum = vec2(0.0);

	float palNoise = (macroVal * wMacroColor) + (mediumVal * wMediumColor) + (microVal * wMicroColor);
	float palT = clamp(0.60 + 0.25 * palNoise + 0.15 * (moisture - 0.5) + 0.10 * (N.y - 0.7), 0.0, 1.0);
	vec3 paletteColor = evalCosinePalette(palA, palB, palC, palD, palT);

	// 1. Climate biome color blending: enhance greenness in moist regions
	if (grassW > 0.01) {
		vec3 lushGreen = mix(paletteColor, vec3(0.12, 0.48, 0.10), smoothstep(0.35, 0.75, moisture) * 0.5);
		finalAlbedo = mix(finalAlbedo, lushGreen, grassW * 0.70);
		roughness = mix(roughness, clamp(wb.roughness + 0.08 * (mediumVal * wMediumColor + microVal * wMicroColor), 0.25, 0.95), grassW * 0.5);
		perturbedGradSum += (macroGrad * wMacroNorm + mediumGrad * wMediumNorm + microGrad * wMicroNorm) * (0.18 * grassW);
	}

	// 2. Patches of green moss / damp vegetation on wet cliff ledges and crevices
	if (rockW > 0.01 && moisture > 0.20 && (wMediumColor > 0.001 || wMacroColor > 0.001)) {
		float ledgeFactor = smoothstep(0.25, 0.75, N.y);
		float mossAmount = rockW * ledgeFactor * smoothstep(0.25, 0.65, moisture) * smoothstep(-0.3, 0.4, mediumVal) * (wMediumColor * 0.7 + wMacroColor * 0.3);
		vec3 mossBase = evalCosinePalette(palA, palB, palC, palD, 0.35 + 0.20 * mediumVal);
		vec3 mossColor = mix(mossBase, vec3(0.10, 0.52, 0.12), moisture * 0.7);
		finalAlbedo = mix(finalAlbedo, mossColor, mossAmount * 0.75);
		roughness = mix(roughness, 0.85, mossAmount * 0.5);
		perturbedGradSum += mediumGrad * (0.15 * mossAmount * wMediumNorm);
	}

	// 3. Procedural Tree Forest Canopy Proxy & Feature-Size Normal Perturbation
	if (hasWeather && grassW > 0.02 && N.y > 0.50 && worldPos.y > params.waterLevel - 0.5) {
		uint biomeIdx = uint(clamp(wb.biomeIndex, 0.0, 9.0));
		uint treeType = getWhittakerTreeType(biomeIdx, temp, moisture);
		float forestDensity = smoothstep(0.25, 0.75, moisture) * smoothstep(0.50, 0.88, N.y);
		vec3 leafColor = vec3(0.18, 0.48, 0.14);

		if (treeType == 3u) {
			forestDensity = smoothstep(0.10, 0.35, moisture) * 0.4 * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.22, 0.45, 0.24);
		} else if (treeType == 2u) {
			forestDensity = smoothstep(0.30, 0.80, moisture) * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.15, 0.55, 0.20);
		} else if (treeType == 0u) {
			forestDensity = smoothstep(0.20, 0.65, moisture) * smoothstep(0.50, 0.88, N.y);
			leafColor = vec3(0.08, 0.30, 0.12);
		}

		if (forestDensity > 0.01) {
			float proxyDistFade = smoothstep(100.0, 1800.0, camDist);
			float proxyAmount = grassW * forestDensity * proxyDistFade;

			vec2 forestGrad = vec2(0.0);
			if (treeType == 0u) {
				forestGrad = macroGrad * (1.3 * 0.50 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.15 * wMicroNorm);
			} else if (treeType == 2u) {
				forestGrad = macroGrad * (0.50 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.20 * wMicroNorm);
			} else if (treeType == 3u) {
				forestGrad = mediumGrad * (0.60 * wMediumNorm) +
				             microGrad * (0.40 * wMicroNorm);
			} else {
				forestGrad = macroGrad * (0.45 * wMacroNorm) +
				             mediumGrad * (0.35 * wMediumNorm) +
				             microGrad * (0.20 * wMicroNorm);
			}

			perturbedGradSum += forestGrad * proxyAmount;

			vec3 canopyHighlight = evalCosinePalette(palA, palB, palC, palD, 0.80 + 0.20 * mediumVal);
			leafColor = mix(leafColor, canopyHighlight, 0.40);
			float crownCrevice = smoothstep(-0.4, 0.4, macroVal * 0.65 + mediumVal * 0.35);
			vec3 canopyColor = mix(leafColor * 0.55, leafColor * 1.15, crownCrevice);

			finalAlbedo = mix(finalAlbedo, canopyColor, proxyAmount * 0.70);
			roughness = mix(roughness, clamp(roughness + 0.22, 0.75, 0.95), proxyAmount);
		}
	}

	// 4. Macro Tier
	if (wMacroColor > 0.001 || wMacroNorm > 0.001) {
		if (rockW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.8 + 0.35 * smoothstep(-0.5, 0.5, macroVal)), rockW * wMacroColor * 0.35);
			perturbedGradSum += macroGrad * 0.35 * rockW * wMacroNorm;
		}
		if (grassW > 0.001) {
			finalAlbedo = mix(finalAlbedo, finalAlbedo * (0.9 + 0.2 * macroVal), grassW * wMacroColor * 0.35);
		}
	}

	// 5. Medium Tier
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

	// 6. Micro Tier
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
	float gradLen = length(perturbedGradSum);
	if (gradLen > 0.45) {
		perturbedGradSum *= 0.45 / gradLen;
	}
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
	// finalAlbedo = getLocalWind(worldPos, uTimeMs);
	// roughness *= 1.0+dot(N, getLocalWind(worldPos, uTimeMs));
	vec3 wind = getLocalWind(worldPos, uTimeMs);

// 	vec2  pXZ      = mod(worldPos.xz, 256.0);
// 	float timeSec  = float(uTimeMs) * 0.001;
// 	float gustWave = 0.5 + 0.5 * sin(dot(pXZ, normalize(wind.xz)) * 0.0135 - timeSec * (0.15 + length(wind) * 0.150)
// 									+ sin(dot(pXZ, vec2(-normalize(wind.xz).y, normalize(wind.xz).x)) * 0.017) * 01.4);
// 	float lean     = gustWave * clamp(length(wind), 0.0, 1.5);

// // vec2 grassSampleXZ = worldPos.xz - dir2D * (lean * 0.18 * bladeHeight);
// 	finalAlbedo *= 1.0 + 0.15 * lean;

	// finalAlbedo = wind;
	vec3 windGrad3D;
	vec2 windValPhase = phasorNoise3DGrad(worldPos*0.0001, smoothstep(0.0, 10.0, length(wind)), wind, N, 0.0, 0.0001*uTime, windGrad3D);
	float windVal = windValPhase.x / max(length(windValPhase), 0.0001);
	// finalAlbedo = windVal * wind;
	finalAlbedo *= 1.0 + 0.15 * clamp(windVal,0,1);
	roughness *= 1.0 - 0.15 * windVal;

/*
// --- Insert before `outPosition = ...` in terrain fragment shader ---
if (params.hydroMapIndex > 0u) {
	int dim = int(params.gridParams.w);
	float worldExtent = 2.0 * FAKE_PLANET_RADIUS * PI;

	// Toroidal UV matching climate_ca's `(pos + dim) % dim` wrapping
	vec2 hydroUV = fract(worldPos.xz / worldExtent + 0.5);
	vec2 texelPos = hydroUV * float(dim) - 0.5;
	ivec2 p00 = (ivec2(floor(texelPos)) + dim) % dim;
	ivec2 p10 = (p00 + ivec2(1, 0)) % dim;
	ivec2 p01 = (p00 + ivec2(0, 1)) % dim;
	ivec2 p11 = (p00 + ivec2(1, 1)) % dim;
	vec2 f = fract(texelPos);

	vec4 h00 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p00);
	vec4 h10 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p10);
	vec4 h01 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p01);
	vec4 h11 = imageLoad(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p11);

	// vec4 h00 = texelFetch(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p00, 0);
	// vec4 h10 = texelFetch(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p10, 0);
	// vec4 h01 = texelFetch(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p01, 0);
	// vec4 h11 = texelFetch(uImagesRGBA32F[nonuniformEXT(params.hydroMapIndex)], p11, 0);
	// vec4 h00 = SAMPLE_2D_NEAREST_FETCH(params.hydroMapIndex, p00, 0);
	// vec4 h10 = SAMPLE_2D_NEAREST_FETCH(params.hydroMapIndex, p10, 0);
	// vec4 h01 = SAMPLE_2D_NEAREST_FETCH(params.hydroMapIndex, p01, 0);
	// vec4 h11 = SAMPLE_2D_NEAREST_FETCH(params.hydroMapIndex, p11, 0);

	// Bilinear interpolation of continuous scalar fields
	float filledElevNorm = mix(mix(h00.r, h10.r, f.x), mix(h01.r, h11.r, f.x), f.y);
	float discharge      = mix(mix(h00.g, h10.g, f.x), mix(h01.g, h11.g, f.x), f.y);
	// finalAlbedo = vec3(0,0,filledElevNorm);

	// Bilinear interpolation of decoded 2D flow vectors (avoids discrete index averaging)
	vec2 v00 = decodeFlowDir(h00.b);
	vec2 v10 = decodeFlowDir(h10.b);
	vec2 v01 = decodeFlowDir(h01.b);
	vec2 v11 = decodeFlowDir(h11.b);
	vec2 flowVec = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
	float flowMag = length(flowVec);
	vec2 flowDir2D = (flowMag > 1e-4) ? (flowVec / flowMag) : vec2(0.0);

	// 1. Lakes / Filled Basins: compare filled elevation (0..1000m) against actual terrain Y
	float filledHeightMeters = filledElevNorm * 1000.0;
	float lakeDepth = filledHeightMeters - worldPos.y;
	bool aboveOcean = worldPos.y > (params.waterLevel + 2.0);

	if (aboveOcean && lakeDepth > 0.5) {
		float depthMask = clamp(lakeDepth / 25.0, 0.25, 0.90);
		// Contour rings every 2 meters to inspect basin spill points
		float contour = smoothstep(0.90, 1.0, abs(fract(lakeDepth * 0.5) * 2.0 - 1.0));
		vec3 lakeColor = mix(vec3(0.04, 0.38, 0.55), vec3(0.01, 0.12, 0.28), depthMask);
		lakeColor = mix(lakeColor, vec3(0.45, 0.85, 0.95), contour * 0.45);

		finalAlbedo = mix(finalAlbedo, lakeColor, 0.78);
		roughness = 0.08;
	}

	// 2. Rivers & Animated Flow Chevrons (log-scaled discharge)
	const float RIVER_VIS_MIN = 1.5;
	if (aboveOcean && discharge > RIVER_VIS_MIN) {
		float logDischarge = clamp(log2(discharge / RIVER_VIS_MIN) / 7.0, 0.0, 1.0);
		vec3 riverColor = mix(vec3(0.10, 0.55, 0.85), vec3(0.05, 0.90, 0.95), logDischarge);

		// Project world XZ onto local flow frame to draw downstream-scrolling chevrons
		if (flowMag > 1e-4) {
			vec2 flowPerp = vec2(-flowDir2D.y, flowDir2D.x);
			float cellSize = worldExtent / float(dim);
			float along = dot(worldPos.xz, flowDir2D) / (cellSize * 0.5);
			float across = abs(fract(dot(worldPos.xz, flowPerp) / (cellSize * 0.5)) - 0.5) * 2.0;

			// V-shaped chevron pointing downstream, animated by uTime
			float chevronPhase = fract(along - across * 0.75 - uTime * (0.4 + logDischarge * 0.8));
			float chevron = smoothstep(0.72, 0.88, chevronPhase) * (1.0 - smoothstep(0.92, 1.0, chevronPhase));
			riverColor = mix(riverColor, vec3(1.0, 0.95, 0.45), chevron * 0.85);
		}

		float riverBlend = mix(0.45, 0.92, logDischarge);
		finalAlbedo = mix(finalAlbedo, riverColor, riverBlend);
		roughness = mix(roughness, 0.12, riverBlend);
	}

	// 3. Diagnostic Alert: inland cell with no valid outflow (incomplete fill convergence)
	if (aboveOcean && flowMag < 1e-4 && discharge > RIVER_VIS_MIN) {
		finalAlbedo = vec3(1.0, 0.0, 0.75);
	}
}
*/

	float dropOff = dot(worldPos.xz, worldPos.xz) / (2.0 * FAKE_PLANET_RADIUS);
	float flatWorldY = worldPos.y + dropOff;

	if (params.hydroMapIndex > 0u && flatWorldY > params.waterLevel) {
		vec2 flowVec;
		vec4 hydro = sampleHydrologySmooth(params.hydroMapIndex, worldPos.xz, flowVec);

		float filledWorldY = hydro.r * 1000.0; // Must match climate_initial_2.comp's elevation scale
		float discharge    = max(0.0, hydro.g);
		float flowMag      = length(flowVec);

		// 1. LAKES & FILLED BASINS (using curvature-corrected flatWorldY)
		float lakeDepth = filledWorldY - flatWorldY;
		if (lakeDepth > 0.25) {
			float depthFactor = clamp(lakeDepth / 25.0, 0.0, 1.0);
			vec3 lakeColor = mix(vec3(0.08, 0.38, 0.48), vec3(0.02, 0.14, 0.26), depthFactor);

			float shoreFoam = 1.0 - smoothstep(0.25, 1.8, lakeDepth);
			lakeColor = mix(lakeColor, vec3(0.75, 0.88, 0.92), shoreFoam * 0.65);

			finalAlbedo = mix(finalAlbedo, lakeColor, 0.85);
			roughness = 0.06;
		}

		// 2. RIVERS & DRAINAGE ACCUMULATION
		float logDischarge = log2(1.0 + discharge * 4.0);
		float riverIntensity = smoothstep(0.25, 3.2, logDischarge);

		if (riverIntensity > 0.01) {
			float edgeBreakup = smoothstep(0.15, 0.55, riverIntensity + warpNoise.x * 0.12);
			vec3 riverColor = mix(vec3(0.10, 0.45, 0.65), vec3(0.04, 0.22, 0.55), riverIntensity);

			// 3. ANIMATED FLOW CHEVRONS (anchored to camera-relative inPosition.xz to avoid float shear)
			if (flowMag > 0.05) {
				vec2 dir = flowVec / flowMag;
				vec2 perp = vec2(-dir.y, dir.x);

				vec2 localFlowUV = fract(vec2(dot(inPosition.xz, perp), dot(inPosition.xz, dir)) / 250.0) - 0.5;
				float chevronCoord = localFlowUV.y + abs(localFlowUV.x) * 1.4 - (uTime * 0.35 * (0.5 + riverIntensity));
				float chevronPulse = smoothstep(0.35, 0.5, abs(fract(chevronCoord) - 0.5));

				riverColor = mix(riverColor, vec3(0.55, 0.92, 1.0), chevronPulse * 0.75);
			}

			finalAlbedo = mix(finalAlbedo, riverColor, edgeBreakup * 0.85);
			roughness = mix(roughness, 0.08, edgeBreakup * 0.8);
		}
	}


	outPosition = vec4(inPosition, 0.0);
	outNormal   = vec4(normalize(N), 1.0);
	outAlbedo   = vec4(finalAlbedo, inAlbedo.a > 0.0 ? inAlbedo.a : 1.0);
	outMaterial = vec4(metallic, roughness, glint, inMaterial.a);
}
