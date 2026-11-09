#version 460 core
#include "bindless.glsl"

// Matches particle.mesh's split: particleType is flat (required for an integer interface
// variable), everything else here needs real per-pixel interpolation -- uv especially, since it
// is what the shape functions below actually vary over.
layout(location = 0) in struct ParticleVertexOutput {
	vec4 color;
	vec2 uv;
	vec4 misc;
	vec3 worldPos;
} IN;

layout(location = 4) flat in uint inParticleType;

layout(location = 0) out vec4 outColor;

void main() {
	vec2 p = IN.uv - vec2(0.5);
	float shapeAlpha = 0.0;

	vec3 particleColor = IN.color.rgb;

	if (inParticleType == 0u) { // Bird shape
		float wingFlap = sin(IN.misc.x) * 0.35;
		float wingY = abs(p.x) * wingFlap;
		float distToWing = abs(p.y - wingY);

		float body = smoothstep(0.18, 0.05, length(p * vec2(2.5, 1.0)));
		float wings = smoothstep(0.1, 0.01, distToWing) * step(abs(p.x), 0.45);
		shapeAlpha = max(body, wings);
	} else if (inParticleType == 1u) { // Fish shape
		float tailWiggle = sin(IN.misc.x) * 0.4;
		float tailY = (p.x + 0.1) * tailWiggle;

		float body = smoothstep(0.22, 0.05, length(p * vec2(1.5, 3.2)));
		float tail = smoothstep(0.12, 0.01, abs(p.y - tailY)) * step(-0.45, p.x) * step(p.x, -0.1);
		shapeAlpha = max(body, tail);
	} else { // Firefly shape (Type 2): Small glowing vesica / ellipse
		float vesicaD = max(length(p - vec2(0.08, 0.0)), length(p + vec2(0.08, 0.0)));
		float core = smoothstep(0.38, 0.05, vesicaD);
		float halo = exp(-vesicaD * 4.5) * 0.75;
		shapeAlpha = clamp(core + halo, 0.0, 1.0);

		float pulse = max(0.2, 0.5 + 0.5 * sin(IN.misc.x));
		particleColor *= pulse;
	}

	// Smooth radial edge fade near the quad perimeter (p in [-0.5, 0.5]) to ensure
	// alpha smoothly drops to zero before the quad boundary, eliminating square edge artifacts.
	float edgeFade = smoothstep(0.5, 0.42, length(p));
	shapeAlpha *= edgeFade;

	if (shapeAlpha < 0.005) {
		discard;
	}

	float alpha = clamp(IN.color.a * shapeAlpha, 0.0, 1.0);
	if (alpha < 0.005) {
		discard;
	}

	// McGuire Weighted Order-Independent Translucency / Blending (WOITB) weight
	// Evaluates McGuire & Bavoil (2013) Eq. 10 weight function over camera distance z:
	float z = max(0.1, length(IN.worldPos - uCameraPosition.xyz));
	float weight = clamp(10.0 / (1e-5 + pow(z / 5.0, 2.0) + pow(z / 200.0, 6.0)), 0.01, 3000.0);

	// Weighted color accumulation (Channel 1 / RGB)
	vec3 accumColor = particleColor * alpha * weight;

	// Revealage factor for dual-channel WOITB (transmittance 1.0 - alpha) stored
	// in the available free channel (Alpha channel of HdrColor RGBA16F)
	float revealage = clamp(1.0 - alpha, 0.0, 1.0);

	// Dual-channel WOITB output:
	// RGB: Premultiplied depth-weighted color accumulation
	// A: Revealage (transmittance)
	outColor = vec4(accumColor, revealage);
}
