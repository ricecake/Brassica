#version 460
#include "bindless.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	uvec4 gridParams;     // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = unused
	vec3  waterColor;
	float waterLevel;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  gNormalIndex;
	uint  padding;
} params;

void main() {
	ivec2 gTexSize = textureSize(sampler2D(uTextures2D[nonuniformEXT(params.gAlbedoIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), 0);
	vec2 screenUV = gl_FragCoord.xy / vec2(gTexSize);

	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, screenUV);
	vec3 terrainPos = SAMPLE_NEAREST(params.gPositionIndex, screenUV).rgb;

	float distToCamWater = length(uCameraPosition.xyz - inWorldPos);
	float depthBelowWater = 100.0; // Default deep water depth when background is sky

	if (albedo.a >= 0.01) {
		float distToCamTerrain = length(uCameraPosition.xyz - terrainPos);
		// Terrain is strictly in front of the water mesh fragment
		if (distToCamTerrain < distToCamWater - 0.2) {
			outColor = vec4(0.0);
			return;
		}

		depthBelowWater = inWorldPos.y - terrainPos.y;
		if (depthBelowWater <= 0.0) {
			outColor = vec4(0.0);
			return;
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
	vec3 refractedPos = SAMPLE_NEAREST(params.gPositionIndex, refractUV).rgb;
	if (inWorldPos.y - refractedPos.y <= 0.0 || refractedAlbedo.a < 0.01) {
		refractedAlbedo = albedo;
	}

	// Translucency & Beer-Lambert absorption: shallow water is clearer, deep water absorbs more light
	float absorption = 1.0 - exp(-depthBelowWater * 0.15);
	vec3 deepWaterColor = vec3(0.01, 0.15, 0.35);
	vec3 waterBodyColor = mix(params.waterColor, deepWaterColor, absorption);

	float alpha = clamp(0.35 + absorption * 0.5, 0.35, 0.85);

	// Specular shine and Fresnel reflection when camera is close
	vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
	vec3 viewDir = normalize(uCameraPosition.xyz - inWorldPos);
	vec3 halfDir = normalize(lightDir + viewDir);

	float NdotH = max(dot(waveNormal, halfDir), 0.0);
	float specular = pow(NdotH, 128.0) * closeFactor;
	vec3 shineColor = vec3(1.2, 1.1, 0.9) * specular;

	float fresnel = pow(1.0 - max(dot(viewDir, waveNormal), 0.0), 4.0);
	vec3 finalWaterTint = mix(waterBodyColor, vec3(0.6, 0.8, 1.0), fresnel * 0.4);

	vec3 blendedColor = mix(refractedAlbedo.rgb, finalWaterTint, alpha) + shineColor;
	outColor = vec4(blendedColor, alpha);
}
