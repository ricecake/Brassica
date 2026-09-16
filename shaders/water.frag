#version 460
#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	uvec4 gridParams;    // x = numRings, y = meshletsPerRow, z = totalMeshlets, w = textureDim
	uvec4 lodOffsets0_3;
	uvec4 lodOffsets4_7;
	uvec4 lodOffsets8_11;
	vec3  waterColor;
	float waterLevel;
	uint  clipmapIndex;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  gNormalIndex;
}

params;

void main() {
	ivec2 gTexSize = textureSize(
		sampler2D(uTextures2D[nonuniformEXT(params.gAlbedoIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		0
	);
	vec2 screenUV = gl_FragCoord.xy / vec2(gTexSize);

	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, screenUV);
	vec3 relTerrainPos = SAMPLE_NEAREST(params.gPositionIndex, screenUV).rgb;
	vec3 relWaterPos = inWorldPos - uCameraPosition.xyz;

	float distToCamWater = length(relWaterPos);
	float depthBelowWater = 100.0; // Default deep water depth when background is sky

	if (albedo.a >= 0.01) {
		float distToCamTerrain = length(relTerrainPos);
		// Terrain is strictly in front of the water mesh fragment
		if (distToCamTerrain < distToCamWater - 0.2) {
			discard;
		}

		depthBelowWater = relWaterPos.y - relTerrainPos.y;
		if (depthBelowWater <= 0.0) {
			discard;
		}
	}

	vec3 waveNormal = normalize(inNormal);
	if (length(waveNormal) < 0.1) {
		waveNormal = vec3(0.0, 1.0, 0.0);
	}

	float distToCam = distToCamWater;
	float closeThreshold = 800.0;
	float closeFactor = clamp(1.0 - distToCam / closeThreshold, 0.0, 1.0);

	// Refraction: distort G-Buffer sample UVs based on wave normal and water depth
	vec2 refractOffset = waveNormal.xz * clamp(depthBelowWater * 0.015, 0.0, 0.04) * closeFactor;
	vec2 refractUV = clamp(screenUV + refractOffset, vec2(0.0), vec2(1.0));

	vec4 refractedAlbedo = SAMPLE_NEAREST(params.gAlbedoIndex, refractUV);
	vec3 relRefractedPos = SAMPLE_NEAREST(params.gPositionIndex, refractUV).rgb;
	if (relWaterPos.y - relRefractedPos.y <= 0.0 || refractedAlbedo.a < 0.01) {
		refractedAlbedo = albedo;
	}

	// Translucency & Beer-Lambert absorption: shallow water is clearer turquoise, deep water absorbs more light
	float absorption = 1.0 - exp(-depthBelowWater * 0.12);
	vec3  shallowWaterTint = vec3(0.12, 0.62, 0.78);
	vec3  deepWaterColor = vec3(0.01, 0.12, 0.32);
	vec3  baseTint = mix(shallowWaterTint, params.waterColor, clamp(depthBelowWater / 5.0, 0.0, 1.0));
	vec3  waterBodyColor = mix(baseTint, deepWaterColor, absorption);

	float alpha = clamp(0.35 + absorption * 0.5, 0.35, 0.88);

	// Specular shine and Fresnel reflection when camera is close
	vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
	vec3 viewDir = normalize(-relWaterPos);
	vec3 halfDir = normalize(lightDir + viewDir);

	float NdotH = max(dot(waveNormal, halfDir), 0.0);
	float specular = pow(NdotH, 128.0) * closeFactor;
	vec3  shineColor = vec3(1.2, 1.1, 0.9) * specular;

	float fresnel = pow(1.0 - max(dot(viewDir, waveNormal), 0.0), 4.0);
	vec3  finalWaterTint = mix(waterBodyColor, vec3(0.65, 0.82, 1.0), fresnel * 0.45);

	vec3 blendedColor = mix(refractedAlbedo.rgb, finalWaterTint, alpha) + shineColor;

	// Shoreline foam and wave crest foam
	float shoreFoam = clamp(1.0 - depthBelowWater / 2.2, 0.0, 1.0);
	shoreFoam = pow(shoreFoam, 1.4);
	float foamNoise = InterleavedGradientNoise(inWorldPos.xz * 3.5, int(uTime * 12.0));
	shoreFoam *= 0.65 + 0.35 * foamNoise;

	float crestFactor = clamp((1.0 - waveNormal.y) * 3.5, 0.0, 1.0);
	float crestFoam = crestFactor * closeFactor * (0.5 + 0.5 * sin(uTime * 3.0 + inWorldPos.x * 0.5));

	float totalFoam = clamp(shoreFoam * 1.25 + crestFoam * 0.6, 0.0, 1.0);
	vec3  foamColor = vec3(0.92, 0.96, 1.0);

	blendedColor = mix(blendedColor, foamColor, totalFoam);
	alpha = max(alpha, totalFoam * 0.92);

	outColor = vec4(blendedColor, alpha);
}
