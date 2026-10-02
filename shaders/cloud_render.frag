#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CloudRenderPushConstants {
	uvec4 cascadeSampledIdx; // x, y, z = sampled bindless indices for cascades 0, 1, 2
	vec4  cameraPos;
} push;

void main() {
	// Sample from the 3 volume texture cascades (minimally populated sample)
	vec4 cascade0Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.x, vec3(vUv, 0.5));
	vec4 cascade1Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, vec3(vUv, 0.5));
	vec4 cascade2Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, vec3(vUv, 0.5));

	vec4 cloudVal = cascade0Sample * 0.5 + cascade1Sample * 0.3 + cascade2Sample * 0.2;
	outColor = cloudVal;
}
