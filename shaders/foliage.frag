#version 460
#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inColor;
layout(location = 4) flat in uint inMaterialType;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;

void main() {
	vec3 relPos = inWorldPos - uCameraPosition.xyz;
	float roughness = 0.5;
	if (inMaterialType == 1u) {        // Fern
		roughness = 0.6;
	} else if (inMaterialType == 2u) { // Flower
		roughness = 0.3;
	} else if (inMaterialType == 3u) { // Rock
		roughness = 0.85;
	} else if (inMaterialType == 4u) { // Seaweed
		roughness = 0.25;
	} else if (inMaterialType == 5u) { // Simple Bush
		roughness = 0.6;
	}
	outPosition = vec4(relPos, 1.0);
	outNormal = vec4(normalize(inNormal), roughness);
	outAlbedo = inColor;
}
