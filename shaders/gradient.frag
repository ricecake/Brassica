#version 460
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(inUV.x, inUV.y, 0.5f, 1.0f);
}
