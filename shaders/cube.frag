#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	float fov;
	float aspectRatio;
	uint  frameIndex;
	uint  globalSeed;
	uint  frameRandom;
} ubo;

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;

void main() {
	gPosition = vec4(inPosition, 1.0);
	gNormal = vec4(normalize(inNormal), 0.0);
	gAlbedo = inAlbedo;
}
