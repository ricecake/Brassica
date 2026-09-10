#version 460

layout(location = 0) in vec2 inUV;
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

vec3 getWaterNormal(vec2 worldXZ, float time) {
	float w1 = sin(worldXZ.x * 0.15 + time * 2.0) * 0.1;
	float w2 = cos(worldXZ.y * 0.15 + time * 1.5) * 0.1;
	float w3 = sin((worldXZ.x + worldXZ.y) * 0.08 + time * 2.5) * 0.08;

	vec3 N = vec3(-w1 - w3, 1.0, -w2 - w3);
	return normalize(N);
}

void main() {
	vec4 albedoSample = texture(gAlbedo, inUV);
	vec3 terrainPos = texture(gPosition, inUV).rgb;
	bool hasTerrain = (albedoSample.a > 0.01);

	vec3 camPos = params.cameraPos.xyz;

	vec3 rayDir;
	if (hasTerrain) {
		rayDir = normalize(terrainPos - camPos);
	} else {
		vec4 ndc = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
		mat4 invViewProj = inverse(params.viewProj);
		vec4 worldTarget = invViewProj * ndc;
		worldTarget /= worldTarget.w;
		rayDir = normalize(worldTarget.xyz - camPos);
	}

	float waterHeight = 0.0;
	if (abs(rayDir.y) < 0.0001) {
		discard;
	}

	float tWater = (waterHeight - camPos.y) / rayDir.y;
	if (tWater <= 0.0) {
		discard;
	}

	vec3 waterHitPos = camPos + tWater * rayDir;
	float distToWater = tWater;

	float distToTerrain;
	if (hasTerrain) {
		distToTerrain = length(terrainPos - camPos);
	} else {
		distToTerrain = 1e6;
	}

	if (distToTerrain < distToWater) {
		discard;
	}

	float waterDepth = clamp(distToTerrain - distToWater, 0.0, 100.0);

	vec3 N = getWaterNormal(waterHitPos.xz, ubo.time);
	vec3 V = normalize(camPos - waterHitPos);

	vec3 lightDir = normalize(vec3(0.5, 0.4, 0.5));
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

	float alpha = clamp(0.3 + depthFactor * 0.55 + fresnel * 0.3, 0.2, 0.9);

	vec3 finalColor = waterBaseColor + specularLight;

	outColor = vec4(finalColor, alpha);
}
