#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
	uint  hdrColorIndex;
	float exposure;
	float contrast;
	float saturation;
	float temperature;
	float tint;
	float pMax;
	float pA;
	float pM;
	float pL;
	float pC;
	float pB;
} params;

// Uchimura (Gran Turismo) Tone Mapping Curve Function
// Reference: Hajime Uchimura, "HDR Theory and Practice" (CEDEC 2017)
vec3 uchimura(vec3 x, float P, float a, float m, float l, float c, float b) {
	float l0 = ((P - m) * l) / a;
	float S0 = m + l0;
	float S1 = m + a * l0;
	float C2 = (a * P) / (P - S1);
	float CP = -C2 / P;

	vec3 w0 = vec3(1.0) - smoothstep(vec3(0.0), vec3(m), x);
	vec3 w2 = step(vec3(m + l0), x);
	vec3 w1 = vec3(1.0) - w0 - w2;

	vec3 T = m * pow(max(x / m, vec3(0.0)), vec3(c)) + b;
	vec3 L = m + a * (x - m);
	vec3 S = P - (P - S1) * exp(CP * (x - S0));

	return T * w0 + L * w1 + S * w2;
}

// Convert temperature (-1.0 to 1.0) and tint (-1.0 to 1.0) into RGB white balance scaling
vec3 calculateWhiteBalance(float temp, float tintVal) {
	// Temperature: -1.0 (cool/blue) to +1.0 (warm/orange)
	// Tint: -1.0 (green) to +1.0 (magenta)
	float rScale = 1.0 + temp * 0.2 - tintVal * 0.05;
	float gScale = 1.0 + tintVal * 0.2;
	float bScale = 1.0 - temp * 0.2 - tintVal * 0.05;
	return max(vec3(0.01), vec3(rScale, gScale, bScale));
}

void main() {
	vec3 hdr = SAMPLE_NEAREST(params.hdrColorIndex, inUV).rgb;

	// 1. Exposure
	float expVal = params.exposure > 0.0 ? params.exposure : 1.0;
	hdr *= expVal;

	// 2. White Balance / Temperature & Tint
	hdr *= calculateWhiteBalance(params.temperature, params.tint);

	// 3. Contrast adjustment (in linear space around mid-gray 0.18)
	float contrastVal = params.contrast > 0.0 ? params.contrast : 1.0;
	hdr = max(vec3(0.0), (hdr - vec3(0.18)) * contrastVal + vec3(0.18));

	// 4. Saturation adjustment (Rec. 709 luminance weights)
	float saturationVal = params.saturation >= 0.0 ? params.saturation : 1.0;
	float lum = dot(hdr, vec3(0.2126, 0.7152, 0.0722));
	hdr = max(vec3(0.0), mix(vec3(lum), hdr, saturationVal));

	// 5. Uchimura Tone Mapping
	float P = params.pMax > 0.0 ? params.pMax : 1.0;
	float a = params.pA > 0.0 ? params.pA : 1.0;
	float m = params.pM >= 0.0 ? params.pM : 0.22;
	float l = params.pL >= 0.0 ? params.pL : 0.4;
	float c = params.pC > 0.0 ? params.pC : 1.33;
	float b = params.pB >= 0.0 ? params.pB : 0.0;

	vec3 ldr = uchimura(hdr, P, a, m, l, c, b);

	// 6. Gamma Correction (sRGB gamma ~ 2.2)
	ldr = pow(clamp(ldr, 0.0, 1.0), vec3(1.0 / 2.2));

	outColor = vec4(ldr, 1.0);
}
