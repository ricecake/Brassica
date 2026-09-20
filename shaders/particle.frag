#version 460 core
#include "bindless.glsl"

layout(location = 0) in struct ParticleVertexOutput {
	vec4 color;
	vec2 uv;
	vec4 misc;
	vec3 worldPos;
	flat uint particleType;
} IN;

layout(location = 0) out vec4 outColor;

void main() {
	vec2 p = IN.uv - vec2(0.5);
	float shapeAlpha = 0.0;

	if (IN.particleType == 0u) { // Bird shape
		float wingFlap = sin(IN.misc.x) * 0.35;
		float wingY = abs(p.x) * wingFlap;
		float distToWing = abs(p.y - wingY);

		float body = smoothstep(0.15, 0.05, length(p * vec2(2.5, 1.0)));
		float wings = smoothstep(0.08, 0.01, distToWing) * step(abs(p.x), 0.45);
		shapeAlpha = max(body, wings);
	} else { // Fish shape
		float tailWiggle = sin(IN.misc.x) * 0.4;
		float tailY = (p.x + 0.1) * tailWiggle;

		float body = smoothstep(0.2, 0.05, length(p * vec2(1.2, 2.5)));
		float tail = smoothstep(0.1, 0.01, abs(p.y - tailY)) * step(-0.45, p.x) * step(p.x, -0.1);
		shapeAlpha = max(body, tail);
	}

	if (shapeAlpha < 0.02) {
		discard;
	}

	float alpha = IN.color.a * shapeAlpha;

	// McGuire Weighted Order-Independent Translucency / Blending (WOITB) weight
	float z = length(IN.worldPos - uCameraPosition.xyz);
	float weight = alpha * clamp(10.0 / (1e-5 + pow(z / 200.0, 3.0)), 0.01, 3000.0);

	float woitbAlpha = alpha * clamp(weight / 10.0, 0.1, 1.0);
	outColor = vec4(IN.color.rgb, woitbAlpha);
}
