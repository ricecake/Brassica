#version 460 core
#include "bindless.glsl"

layout(location = 0) in struct ParticleVertexOutput {
	vec4 color;
	vec2 uv;
	vec4 misc;
} IN;

layout(location = 0) out vec4 outColor;

void main() {
	vec2 centerOffset = IN.uv - vec2(0.5);
	float distSq = dot(centerOffset, centerOffset);
	if (distSq > 0.25) {
		discard;
	}

	float alphaFactor = smoothstep(0.25, 0.15, distSq);
	outColor = vec4(IN.color.rgb, IN.color.a * alphaFactor);
}
