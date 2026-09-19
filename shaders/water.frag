#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "lighting.glsl"
#include "clustered_lighting.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	uvec4 gridParams; // x = numRings, y = meshletsPerRow, z = totalMeshlets, w = unused
	vec3  waterColor;
	float waterLevel;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  gNormalIndex;
	uint  padding;
}

params;

// ACES Filmic Tone Mapping Curve (matches deferred.frag)
vec3 ACESFilm(vec3 x) {
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
	ivec2 gTexSize = textureSize(
		sampler2D(uTextures2D[nonuniformEXT(params.gPositionIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		0
	);
	vec2 screenUV = gl_FragCoord.xy / vec2(gTexSize);

	vec4 rawAlbedo = SAMPLE_NEAREST(params.gAlbedoIndex, screenUV);
	vec3 relTerrainPos = SAMPLE_NEAREST(params.gPositionIndex, screenUV).rgb;
	vec3 relWaterPos = inWorldPos - uCameraPosition.xyz;

	float distToCamWater = length(relWaterPos);
	float rayLengthThroughWater = 100.0; // Default deep water ray length when background is sky
	float depthBelowWater = 100.0;       // Vertical Y depth below water

	float distToCamTerrain = length(relTerrainPos);
	if (rawAlbedo.a >= 0.01 || distToCamTerrain > 0.01) {
		// Terrain is strictly in front of the water mesh fragment
		if (distToCamTerrain < distToCamWater - 0.2) {
			outColor = vec4(0.0);
			return;
		}

		rayLengthThroughWater = max(0.0, distToCamTerrain - distToCamWater);
		depthBelowWater = relWaterPos.y - relTerrainPos.y;
		if (depthBelowWater <= 0.0 && rayLengthThroughWater <= 0.0) {
			outColor = vec4(0.0);
			return;
		}
	}

	// Base surface normal
	vec3 baseNormal = normalize(inNormal);
	if (length(baseNormal) < 0.1) {
		baseNormal = vec3(0.0, 1.0, 0.0);
	}

	// Animated multi-frequency sine wave normal perturbations for wave motion across all distances
	vec2 waveXZ = inWorldPos.xz;
	float t = uTime * smoothstep(800.0, 1800.0, distToCamWater) * max(1.0, 1000.0/distToCamWater);

	vec2 w1 = waveXZ * 0.06 + vec2(t * 0.8, t * 0.5);
	vec2 w2 = waveXZ * 0.15 + vec2(-t * 0.7, t * 1.1);
	vec2 w3 = waveXZ * 0.35 + vec2(t * 1.3, -t * 0.9);

	vec2 sinGrad = vec2(
		cos(w1.x + w1.y * 1.2) * 0.18 + cos(w2.x) * 0.10 + sin(w3.x - w3.y) * 0.05,
		sin(w1.y - w1.x * 0.8) * 0.18 + sin(w2.y) * 0.10 + cos(w3.y + w3.x) * 0.05
	);
	vec2 noiseGrad = 0.25 * cross_noise_fbm(inWorldPos * 0.01, 2.0, t).xz;
	sinGrad += noiseGrad * smoothstep(1000.0, 2000.0, distToCamWater) * (1.0 - smoothstep(3000.0, 20000.0, distToCamWater));

	vec3 waveNormal = normalize(baseNormal + vec3(-sinGrad.x, 0.0, -sinGrad.y));

	float distToCam = distToCamWater;
	float closeThreshold = 800.0;
	float closeFactor = clamp(1.0 - distToCam / closeThreshold, 0.0, 1.0);

	// Wave distortion refraction: distort G-Buffer sample UVs based on wave normal & optical depth
	vec2 refractOffset = waveNormal.xz * clamp(rayLengthThroughWater * 0.008, 0.0, 0.03);
	vec2 refractUV = clamp(screenUV + refractOffset, vec2(0.0), vec2(1.0));

	vec4 refrAlbedo = SAMPLE_NEAREST(params.gAlbedoIndex, refractUV);
	vec3 refrNorm = SAMPLE_NEAREST(params.gNormalIndex, refractUV).rgb;
	vec3 refrRelPos = SAMPLE_NEAREST(params.gPositionIndex, refractUV).rgb;

	// Fallback to unrefracted sample if refracted sample hits above water
	if (refrAlbedo.a < 0.01 || relWaterPos.y - refrRelPos.y <= 0.0) {
		refrAlbedo = rawAlbedo;
		refrNorm = SAMPLE_NEAREST(params.gNormalIndex, screenUV).rgb;
		refrRelPos = relTerrainPos;
	}

	// Evaluate HDR lighting for the refracted terrain surface
	vec3 refrWorldPos = refrRelPos + uCameraPosition.xyz;
	vec3 terrainLightContrib = evaluateClusteredLightContribution(refrWorldPos, normalize(refrNorm));
	vec3 litRefractedTerrain = refrAlbedo.rgb * terrainLightContrib;

	// Translucency & Beer-Lambert absorption along view ray path length through water
	vec3 extinctionCoeff = vec3(0.28, 0.07, 0.02);
	vec3 transmittance = exp(-extinctionCoeff * rayLengthThroughWater);

	vec3 shallowWaterTint = vec3(0.12, 0.62, 0.78);
	vec3 deepWaterColor = vec3(0.01, 0.12, 0.32);
	vec3 baseTint = mix(shallowWaterTint, params.waterColor, clamp(depthBelowWater / 5.0, 0.0, 1.0));
	vec3 waterBodyColor = mix(deepWaterColor, baseTint, transmittance);

	// Blend lit refracted terrain with water body optical absorption
	vec3 underwaterSceneHDR = mix(waterBodyColor, litRefractedTerrain, transmittance);

	// Active directional sun light for specular reflections
	vec3 sunDir = normalize(vec3(0.5, 0.8, 0.5));
	vec3 sunColor = vec3(1.2, 1.1, 0.9);
	float sunIntensity = 1.0;
	for (uint i = 0u; i < min(uLightCount, 2u); ++i) {
		if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL && uLights[i].intensity > 0.0) {
			sunDir = normalize(-uLights[i].direction);
			sunColor = uLights[i].color;
			sunIntensity = uLights[i].intensity;
			break;
		}
	}

	// Specular sun glare
	vec3 viewDir = normalize(-relWaterPos);
	vec3 halfDir = normalize(sunDir + viewDir);

	float NdotH = max(dot(waveNormal, halfDir), 0.0);
	float specSharp = pow(NdotH, 256.0);
	float specBloom = pow(NdotH, 20.0) * 0.15;
	float specular = specSharp + specBloom;
	vec3 shineColor = sunColor * sunIntensity * specular * 1.5;

	// Schlick's Fresnel reflection
	float NdotV = max(dot(viewDir, waveNormal), 0.0);
	float fresnel = clamp(pow(1.0 - NdotV, 5.0), 0.02, 0.98);
	vec3 hdrWaterColor = mix(underwaterSceneHDR, vec3(0.65, 0.82, 1.0), fresnel * 0.5) + shineColor;

	// Shoreline foam & wave crest foam
	float shoreFoam = clamp(1.0 - depthBelowWater / 2.2, 0.0, 1.0);
	shoreFoam = pow(shoreFoam, 1.4);
	float foamNoise = InterleavedGradientNoise(inWorldPos.xz * 3.5, int(uTime * 12.0));
	shoreFoam *= 0.65 + 0.35 * foamNoise;

	float crestFactor = clamp((1.0 - waveNormal.y) * 3.5, 0.0, 1.0);
	float crestFoam = crestFactor * closeFactor * (0.5 + 0.5 * sin(uTime * 3.0 + inWorldPos.x * 0.5));

	float totalFoam = clamp(shoreFoam * 1.25 + crestFoam * 0.6, 0.0, 1.0);
	vec3 foamColor = vec3(0.92, 0.96, 1.0);

	hdrWaterColor = mix(hdrWaterColor, foamColor, totalFoam);

	// Tonemap & Gamma correction (matches deferred.frag so water output matches scene brightness)
	vec3 ldrColor = ACESFilm(hdrWaterColor);
	ldrColor = pow(ldrColor, vec3(1.0 / 2.2));

	outColor = vec4(ldrColor, 1.0);
}
