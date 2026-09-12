#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(step(inUV.y, 0.5), step(0.5, inUV.y), 1.0, 1.0f);
}
