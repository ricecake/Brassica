#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	vec4  cameraPos; // xyz = camera position, w = time
	vec3  waterColor;
	float waterLevel;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  gNormalIndex;
	uint  padding;
} params;

void main() {
	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, inUV);
	// albedo.a < 0.01 is the same "no terrain rendered here" sentinel deferred.frag reads --
	// without this check, GBufferPosition's cleared-to-zero value at sky/background pixels
	// would read as "at the water level", tinting the sky.
	if (albedo.a < 0.01) {
		outColor = vec4(0.0);
		return;
	}

	vec3 terrainPos = SAMPLE_NEAREST(params.gPositionIndex, inUV).rgb;
	float depthBelowWater = params.waterLevel - terrainPos.y;
	if (depthBelowWater <= 0.0) {
		outColor = vec4(0.0);
		return;
	}

	vec3 waterWorldPos = vec3(terrainPos.x, params.waterLevel, terrainPos.z);
	float distToCam = length(params.cameraPos.xyz - waterWorldPos);
	float closeThreshold = 400.0;
	float closeFactor = clamp(1.0 - distToCam / closeThreshold, 0.0, 1.0);

	// Wave normal animation based on time (params.cameraPos.w) and world XZ coordinates
	float time = params.cameraPos.w;
	vec2 worldXZ = terrainPos.xz;

	vec2 waveDir1 = vec2(0.8, 0.6);
	vec2 waveDir2 = vec2(-0.5, 0.85);
	vec2 waveDir3 = vec2(0.3, -0.9);

	vec2 waveGrad = waveDir1 * cos(dot(worldXZ, waveDir1) * 0.6 + time * 2.5) * 0.6
	              - waveDir2 * sin(dot(worldXZ, waveDir2) * 1.1 + time * 1.8) * 1.1
	              + waveDir3 * cos(dot(worldXZ, waveDir3) * 2.2 + time * 3.2) * 2.2;

	vec3 waveNormal = normalize(vec3(-waveGrad.x * 0.12 * closeFactor, 1.0, -waveGrad.y * 0.12 * closeFactor));

	// Refraction: distort G-Buffer sample UVs based on wave normal and water depth
	vec2 refractOffset = waveNormal.xz * clamp(depthBelowWater * 0.015, 0.0, 0.03) * closeFactor;
	vec2 refractUV = clamp(inUV + refractOffset, vec2(0.0), vec2(1.0));

	vec4 refractedAlbedo = SAMPLE_NEAREST(params.gAlbedoIndex, refractUV);
	vec3 refractedPos = SAMPLE_NEAREST(params.gPositionIndex, refractUV).rgb;
	if (params.waterLevel - refractedPos.y <= 0.0 || refractedAlbedo.a < 0.01) {
		refractedAlbedo = albedo;
	}

	// Translucency & Beer-Lambert absorption: shallow water is clearer, deep water absorbs more light
	float absorption = 1.0 - exp(-depthBelowWater * 0.15);
	vec3 deepWaterColor = vec3(0.01, 0.15, 0.35);
	vec3 waterBodyColor = mix(params.waterColor, deepWaterColor, absorption);

	float alpha = clamp(0.35 + absorption * 0.5, 0.35, 0.85);

	// Specular shine and Fresnel reflection when camera is close
	vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
	vec3 viewDir = normalize(params.cameraPos.xyz - waterWorldPos);
	vec3 halfDir = normalize(lightDir + viewDir);

	float NdotH = max(dot(waveNormal, halfDir), 0.0);
	float specular = pow(NdotH, 128.0) * closeFactor;
	vec3 shineColor = vec3(1.2, 1.1, 0.9) * specular;

	float fresnel = pow(1.0 - max(dot(viewDir, waveNormal), 0.0), 4.0);
	vec3 finalWaterTint = mix(waterBodyColor, vec3(0.6, 0.8, 1.0), fresnel * 0.4);

	vec3 blendedColor = mix(refractedAlbedo.rgb, finalWaterTint, alpha) + shineColor;
	outColor = vec4(blendedColor, alpha);
}
