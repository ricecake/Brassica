#version 460

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inNormal;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	float fov;
	float aspectRatio;
	uint  frameIndex;
	uint  globalSeed;
	uint  frameRandom;
} ubo;

void main() {
	vec3 lightDir = normalize(vec3(sin(ubo.time), sin(ubo.time * 0.5)*0.5+0.5, -cos(ubo.time)));
	vec3 norm = normalize(inNormal);
	float diff = max(dot(norm, lightDir), 0.1);

	outColor = vec4(inColor.rgb * diff, 1.0);
}
