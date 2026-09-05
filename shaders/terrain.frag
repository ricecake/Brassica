#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inAlbedo;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;

void main() {
	outPosition = vec4(inPosition, 1.0);
	outNormal = vec4(normalize(inNormal), 0.0);
	outAlbedo = inAlbedo;
}
