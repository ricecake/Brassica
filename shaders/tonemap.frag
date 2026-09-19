#version 460
#include "bindless.glsl"
#include "helpers/tonemapping.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
	uint  hdrColorIndex;
	uint  bloomTextureIndex;
	uint  toneMapMode;
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
	float bloomIntensity;
	float cdlSaturation;

	vec4 cdlSlope;
	vec4 cdlOffset;
	vec4 cdlPower;
} params;

// Planckian locus approximation for temperature to RGB
vec3 tempToRgb(float temp) {
	if (abs(temp) < 0.001) {
		return vec3(1.0);
	}
	// Temperature in Kelvin scale offset around 6500K
	float kelvin = clamp(6500.0 + temp * 3000.0, 2000.0, 12000.0) / 100.0;
	vec3 rgb;

	if (kelvin <= 66.0) {
		rgb.r = 255.0;
		rgb.g = clamp(99.4708025861 * log(kelvin) - 161.1195681661, 0.0, 255.0);
		if (kelvin <= 19.0) {
			rgb.b = 0.0;
		} else {
			rgb.b = clamp(138.5177312231 * log(kelvin - 10.0) - 305.0447927307, 0.0, 255.0);
		}
	} else {
		rgb.r = clamp(329.698727446 * pow(kelvin - 60.0, -0.1332047592), 0.0, 255.0);
		rgb.g = clamp(288.1221695283 * pow(kelvin - 60.0, -0.0755148492), 0.0, 255.0);
		rgb.b = 255.0;
	}

	return rgb / 255.0;
}

// Convert temperature (-1.0 to 1.0) and tint (-1.0 to 1.0) into RGB white balance gain
vec3 calculateWhiteBalanceGain(float temp, float tintVal) {
	vec3 whiteGain = 1.0 / max(tempToRgb(temp), vec3(0.0001));
	whiteGain.g *= (1.0 - tintVal * 0.1);
	whiteGain.rb *= (1.0 + tintVal * 0.05);

	// Preserve luminance
	float luma = dot(whiteGain, vec3(0.2126, 0.7152, 0.0722));
	return whiteGain / max(luma, 0.0001);
}

void main() {
	vec3 result = SAMPLE_NEAREST(params.hdrColorIndex, inUV).rgb;

	// Add Bloom
	if (params.bloomTextureIndex > 0u && params.bloomIntensity > 0.0) {
		vec3 bloomColor = SAMPLE_LINEAR(params.bloomTextureIndex, inUV).rgb;
		result += bloomColor * params.bloomIntensity;
	}

	// 1. Exposure
	float expVal = params.exposure > 0.0 ? params.exposure : 1.0;
	result *= expVal;

	// 2. White Balance
	result *= calculateWhiteBalanceGain(params.temperature, params.tint);

	// 3. Contrast adjustment (in linear space around mid-gray 0.18)
	float contrastVal = params.contrast > 0.0 ? params.contrast : 1.0;
	if (contrastVal != 1.0) {
		result = max(vec3(0.0), (result - vec3(0.18)) * contrastVal + vec3(0.18));
	}

	// 4. ASC CDL Color Grading
	vec3 slope = params.cdlSlope.rgb != vec3(0.0) ? params.cdlSlope.rgb : vec3(1.0);
	vec3 offset = params.cdlOffset.rgb;
	vec3 power = params.cdlPower.rgb != vec3(0.0) ? params.cdlPower.rgb : vec3(1.0);
	result = pow(max(result * slope + offset, vec3(0.0)), power);

	// 5. Tonemapping
	int mode = int(params.toneMapMode);
	if (mode == 5) { // Uchimura
		float P = params.pMax > 0.0 ? params.pMax : 1.0;
		float a = params.pA > 0.0 ? params.pA : 1.0;
		float m = params.pM >= 0.0 ? params.pM : 0.22;
		float l = params.pL >= 0.0 ? params.pL : 0.4;
		float c = params.pC > 0.0 ? params.pC : 1.33;
		float b = params.pB >= 0.0 ? params.pB : 0.0;
		result = uchimura(result, P, a, m, l, c, b);
	} else {
		result = applyTonemapping(result, mode);
	}

	// 6. Saturation adjustment
	float satVal = params.saturation >= 0.0 ? params.saturation : (params.cdlSaturation >= 0.0 ? params.cdlSaturation : 1.0);
	float luma = dot(result, vec3(0.2126, 0.7152, 0.0722));
	result = max(vec3(0.0), mix(vec3(luma), result, satVal));

	// 7. Gamma Correction (sRGB gamma ~ 2.2)
	vec3 ldr = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));

	outColor = vec4(ldr, 1.0);
}
