#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;
layout(location = 3) in vec4 inMaterial;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

void main() {
	outPosition = vec4(inPosition, 0.0);
	outNormal = vec4(normalize(inNormal), 1.0);
	outAlbedo = inAlbedo;
	outMaterial = inMaterial;
}
