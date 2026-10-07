#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;

#include "bindless.glsl"

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;
layout(location = 3) out vec4 gMaterial;

void main() {
	gPosition = vec4(inPosition, 0.0);
	gNormal = vec4(normalize(inNormal), 1.0);
	gAlbedo = inAlbedo;
	gMaterial = vec4(0.10, 0.35, 0.0, 0.0);
}
