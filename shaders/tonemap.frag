#version 460
#include "bindless.glsl"
#include "helpers/tonemapping.glsl"
#include "types/autoexposure.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
	uint  hdrColorIndex;
	uint  bloomBlurIndex;
	uint  ltmFusedIndex;
	uint  ltmExpMipIndex;

	uint  depthTextureIndex;
	uint  toneMapMode;
	vec2  ltmRes;

	float intensity;
	float minIntensity;
	float maxIntensity;
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

// Calculates a safe multiplier to prevent sky luminance from blowing out the Uchimura shoulder
float calculateSkyAttenuation(vec3 rawHdrColor, float uchimuraM, float uchimuraL, float rolloffStrength) {
	float luma = dot(rawHdrColor, vec3(0.2126, 0.7152, 0.0722));
	float shoulderStart = uchimuraM + uchimuraL;
	float overdrive = max(0.0, luma - shoulderStart);
	float multiplier = 1.0 / (1.0 + rolloffStrength * overdrive);
	return multiplier;
}

void main() {
	vec3 sceneColor = SAMPLE_NEAREST(params.hdrColorIndex, inUV).rgb;
	vec3 bloomColor = vec3(0.0);
	if (params.bloomBlurIndex > 0u) {
		bloomColor = SAMPLE_LINEAR(params.bloomBlurIndex, inUV).rgb;
	}

	vec3 result = sceneColor;

	float rawDepth = 0.0;
	if (params.depthTextureIndex > 0u) {
		rawDepth = SAMPLE_NEAREST(params.depthTextureIndex, inUV).r;
	}

	int isSky = 0;
	if (rawDepth > 0.99999) {
		isSky = 1;
	}

	// 1. Guided Upsampling for LTM (Scene layer only)
	if (layers[0].ltmEnabled != 0 && isSky == 0 && params.ltmFusedIndex > 0u && params.ltmExpMipIndex > 0u && params.ltmRes.x > 0.0 && params.ltmRes.y > 0.0) {
		float autoExp = layers[0].targetLuminance / max(layers[0].avgLuma, 0.0001);
		vec3 currentExposure = result * autoExp;
		vec3 currentAces = aces(currentExposure);
		float guidanceLuma = sqrt(max(dot(currentAces, vec3(0.2126, 0.7152, 0.0722)), 0.0));

		// Sample 3x3 neighborhood from low-res textures for guided filter
		float momentX = 0.0;
		float momentY = 0.0;
		float momentX2 = 0.0;
		float momentXY = 0.0;
		float weightSum = 0.0;

		vec2 texelSize = 1.0 / params.ltmRes;

		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				vec2 offset = vec2(dx, dy) * texelSize;
				float x = SAMPLE_LINEAR(params.ltmExpMipIndex, inUV + offset).y; // mid-exposure lightness
				float y = SAMPLE_LINEAR(params.ltmFusedIndex, inUV + offset).r;  // fused lightness

				float w = exp(-0.5 * float(dx*dx + dy*dy) / (0.7 * 0.7));
				momentX += x * w;
				momentY += y * w;
				momentX2 += x * x * w;
				momentXY += x * y * w;
				weightSum += w;
			}
		}

		momentX /= weightSum;
		momentY /= weightSum;
		momentX2 /= weightSum;
		momentXY /= weightSum;

		float A = (momentXY - momentX * momentY) / (max(momentX2 - momentX * momentX, 0.0) + 0.00001);
		float B = momentY - A * momentX;

		float localFusedLuma = max(A * guidanceLuma + B, 0.0);
		float finalMultiplier = localFusedLuma / max(guidanceLuma, 0.0001);

		// Prevent artifacts in very dark areas
		float lerpToUnityThreshold = 0.007;
		if (guidanceLuma < lerpToUnityThreshold) {
			float t = guidanceLuma / lerpToUnityThreshold;
			finalMultiplier = mix(1.0, finalMultiplier, t * t);
		}

		result *= finalMultiplier;
	}

	// 2. Exposure Application
	if (layers[isSky].useAutoExposure != 0) {
		float autoExposure = layers[isSky].targetLuminance / max(layers[isSky].adaptedLuminance, 0.0001);
		autoExposure = clamp(autoExposure, layers[isSky].minExposure, layers[isSky].maxExposure);

		if (isSky == 1) {
			vec2 ndc = inUV * 2.0 - 1.0;
			vec4 ray_view = uInvProjMatrix * vec4(ndc, -1.0, 1.0);
			ray_view = vec4(ray_view.xy, -1.0, 0.0);
			vec3 worldDir = normalize((uInvViewMatrix * ray_view).xyz);

			float attenuation = calculateSkyAttenuation(result * autoExposure, layers[1].autoUchimuraM, layers[1].autoUchimuraL, 1.20);
			float mask = smoothstep(-3.14 * 0.125, 0.5 * 1.5707, asin(worldDir.y));
			attenuation = mix(attenuation, 1.0, mask);
			autoExposure *= attenuation;
		}

		result *= autoExposure;
	} else {
		float expVal = params.exposure > 0.0 ? params.exposure : 1.0;
		result *= expVal;
	}

	// Accumulate Bloom
	float activeIntensity = params.intensity > 0.0 ? params.intensity : params.bloomIntensity;
	result += bloomColor * activeIntensity;

	// 3. White Balance
	float activeTemp = layers[isSky].whiteTemp != 0.0 ? (layers[isSky].whiteTemp - 6500.0) / 3000.0 : params.temperature;
	float activeTint = layers[isSky].whiteTint != 0.0 ? layers[isSky].whiteTint : params.tint;
	result *= calculateWhiteBalanceGain(activeTemp, activeTint);

	// 4. Contrast adjustment
	float contrastVal = params.contrast > 0.0 ? params.contrast : 1.0;
	if (contrastVal != 1.0) {
		result = max(vec3(0.0), (result - vec3(0.18)) * contrastVal + vec3(0.18));
	}

	// 5. ASC CDL Color Grading
	vec3 slope = layers[isSky].cdlSlope.rgb != vec3(0.0) ? layers[isSky].cdlSlope.rgb : (params.cdlSlope.rgb != vec3(0.0) ? params.cdlSlope.rgb : vec3(1.0));
	vec3 offset = layers[isSky].cdlOffset.rgb != vec3(0.0) ? layers[isSky].cdlOffset.rgb : params.cdlOffset.rgb;
	vec3 power = layers[isSky].cdlPower.rgb != vec3(0.0) ? layers[isSky].cdlPower.rgb : (params.cdlPower.rgb != vec3(0.0) ? params.cdlPower.rgb : vec3(1.0));
	result = pow(max(result * slope + offset, vec3(0.0)), power);

	// 6. Tonemapping
	if (layers[isSky].toneMappingEnabled != 0) {
		int mode = layers[isSky].toneMapMode;
		if (mode == 5) { // Uchimura
			if (layers[isSky].autoTuneEnabled != 0) {
				result = uchimura(result, layers[isSky].autoUchimuraP, layers[isSky].autoUchimuraA, layers[isSky].autoUchimuraM, layers[isSky].autoUchimuraL, layers[isSky].autoUchimuraC, layers[isSky].autoUchimuraB);
			} else {
				result = uchimura(result, layers[isSky].uchimuraP, layers[isSky].uchimuraA, layers[isSky].uchimuraM, layers[isSky].uchimuraL, layers[isSky].uchimuraC, layers[isSky].uchimuraB);
			}
		} else {
			result = applyTonemapping(result, mode);
		}
	} else {
		int mode = int(params.toneMapMode);
		if (mode == 5) { // Uchimura
			float P = params.pMax > 0.0 ? params.pMax : 1.0;
			float a = params.pA > 0.0 ? params.pA : 1.0;
			float m = params.pM >= 0.0 ? params.pM : 0.22;
			float l = params.pL >= 0.0 ? params.pL : 0.4;
			float c = params.pC > 0.0 ? params.pC : 1.33;
			float b = params.pB >= 0.0 ? params.pB : 0.0;
			result = uchimura(result, P, a, m, l, c, b);
		} else if (mode != 9) {
			result = applyTonemapping(result, mode);
		}
	}

	// 7. Saturation adjustment
	float satVal = layers[isSky].cdlSaturation > 0.0 ? layers[isSky].cdlSaturation : (params.saturation >= 0.0 ? params.saturation : (params.cdlSaturation >= 0.0 ? params.cdlSaturation : 1.0));
	float luma = dot(result, vec3(0.2126, 0.7152, 0.0722));
	result = max(vec3(0.0), mix(vec3(luma), result, satVal));

	// 8. Gamma Correction (sRGB gamma ~ 2.2)
	vec3 ldr = pow(clamp(result, 0.0, 1.0), vec3(1.0 / 2.2));

	outColor = vec4(ldr, 1.0);
}
