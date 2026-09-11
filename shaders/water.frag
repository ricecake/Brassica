#version 460

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	uint  frameIndex;
	uint  globalSeed;
	uint  frameRandom;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D gPosition;
layout(set = 1, binding = 1) uniform sampler2D gDepth;
layout(set = 1, binding = 2) uniform sampler2D gAlbedo;
layout(set = 1, binding = 3) uniform sampler2DArray terrainClipmap;

layout(push_constant) uniform TerrainPushConstants {
	mat4  viewProj;
	vec4  cameraPos;
	uvec4 gridParams;
	uvec4 lodOffsets0_3;
	uvec4 lodOffsets4_7;
} params;

// Fine detail normal perturbation for water surface
vec3 getFineWaterNormal(vec2 worldXZ, float time, vec3 baseNormal) {
	float w1 = sin(worldXZ.x * 0.4 + time * 3.0) * 0.05;
	float w2 = cos(worldXZ.y * 0.4 + time * 2.5) * 0.05;
	float w3 = sin((worldXZ.x + worldXZ.y) * 0.2 + time * 4.0) * 0.03;

	vec3 N = baseNormal + vec3(-w1 - w3, 0.0, -w2 - w3);
	return normalize(N);
}

void main() {
	ivec2 texDim = textureSize(gPosition, 0);
	vec2 screenUV = (texDim.x > 0 && texDim.y > 0) ? (gl_FragCoord.xy / vec2(texDim)) : inUV;

	vec4 albedoSample = texture(gAlbedo, screenUV);
	vec3 terrainPos = texture(gPosition, screenUV).rgb;
	bool hasTerrain = (albedoSample.a > 0.01);

	vec3 camPos = params.cameraPos.xyz;
	vec3 waterHitPos = inWorldPos;

	float distToWater = length(waterHitPos - camPos);

	float distToTerrain;
	if (hasTerrain) {
		distToTerrain = length(terrainPos - camPos);
	} else {
		distToTerrain = 1e6;
	}

	// Discard if terrain is strictly in front of water (with 0.1 margin for depth precision)
	if (hasTerrain && distToTerrain < distToWater - 0.1) {
		discard;
	}

	float waterDepth = clamp(distToTerrain - distToWater, 0.0, 100.0);

	vec3 N = getFineWaterNormal(waterHitPos.xz, ubo.time, inNormal);
	vec3 V = normalize(camPos - waterHitPos);

	vec3 lightDir = normalize(vec3(0.5, 0.8, 0.5));
	vec3 sunColor = vec3(2.5, 2.3, 2.0);

	float F0 = 0.02;
	float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, N), 0.0), 5.0);

	vec3 H = normalize(lightDir + V);
	float spec = pow(max(dot(N, H), 0.0), 128.0);
	vec3 specularLight = sunColor * spec * 1.5;

	vec3 shallowColor = vec3(0.1, 0.6, 0.7);
	vec3 deepColor    = vec3(0.02, 0.08, 0.3);

	float depthFactor = 1.0 - exp(-waterDepth * 0.08);
	vec3 waterBaseColor = mix(shallowColor, deepColor, depthFactor);

	float alpha = clamp(0.35 + depthFactor * 0.5 + fresnel * 0.3, 0.25, 0.95);

	vec3 finalColor = waterBaseColor + specularLight;

	outColor = vec4(finalColor, alpha);
}
