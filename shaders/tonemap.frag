#version 460
#include "bindless.glsl"
#include "helpers/tonemapping.glsl"
#include "types/autoexposure.glsl"
#include "types/cdl_grading.glsl"

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
	float _pad0; // see TonemapPushConstants.hpp's comment on this field

	float contrast;
	// saturation is no longer read: the old sky-only saturation step it fed is gone now that ASC
	// CDL grading (CdlGradingLayers, step 5 below) always runs and already carries its own
	// per-entry saturation uniformly. Kept as inert padding rather than removed outright -- same
	// reasoning as the `exposure` field's comment above (deleting a scalar here shifts every
	// field after it in C++ with no compensating padding).
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
	// cdlSaturation/cdlSlope/cdlOffset/cdlPower below are no longer read, same reasoning as
	// `saturation` just above -- they fed the old isSky==1 flat-CDL branch, which CdlGradingLayers
	// now uniformly subsumes for every pixel.
	float cdlSaturation;

	vec4 cdlSlope;
	vec4 cdlOffset;
	vec4 cdlPower;

	// Appended last -- see TonemapPushConstants.hpp's comment on why this field specifically has
	// to go here and nowhere else in this struct.
	int  bloomEnabled;

	// Same reasoning -- appended after bloomEnabled, another trailing scalar with nothing after
	// it needing bigger alignment.
	int  numCdlEntries;
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

// Simulates the scotopic rod shift in mesopic lighting conditions
vec3 ApplyPurkinjeShift(vec3 exposedLinearColor, float avgLuminance, vec3 scotopicTint) {
	float scotopicMin = 0.001;
	float photopicMax = 3.0;

	// Calculate logarithmic blend factor
	float logAvg = log(max(avgLuminance, 1e-5));
	float logMin = log(scotopicMin);
	float logMax = log(photopicMax);

	float photopicWeight = clamp((logAvg - logMin) / (logMax - logMin), 0.0, 1.0);
	photopicWeight = smoothstep(0.0, 1.0, photopicWeight);

	// Calculate scene luminance using Rec. 709 luma
	float pixelLuminance = dot(exposedLinearColor, vec3(0.2126, 0.7152, 0.0722));
	vec3 scotopicColor = pixelLuminance * scotopicTint;

	return mix(scotopicColor, exposedLinearColor, photopicWeight);
}

// Steps 2-7 of the old per-isSky pipeline (exposure -> Purkinje -> bloom accumulate -> white
// balance -> contrast -> CDL -> tonemap -> gamma), run once per active ExposureLayer and blended
// by the caller via the same sequential mix() idiom CdlGradingLayers' own entries already use.
// CDL must run here, before tonemapping and on linear HDR data, same position it held in the old
// single-layer pipeline -- it can't move to after the per-layer blend below since each layer may
// run a different tonemap curve. CdlGradingLayers' depth-ranged entries now apply uniformly
// (every layer runs the same CDL loop) instead of being gated on isSky == 0 the way the old flat
// per-layer CDL fields were.
vec3 ApplyExposureLayerChain(ExposureLayer layer, vec3 ltmCorrectedColor, vec3 bloomColor, bool isSkyPixel, float linearZ) {
	vec3 result = ltmCorrectedColor;

	// 2. Exposure Application
	if (layer.data.useAutoExposure != 0) {
		float autoExposure = layer.data.targetLuminance / max(layer.data.adaptedLuminance, 0.0001);
		autoExposure = clamp(autoExposure, layer.data.minExposure, layer.data.maxExposure);
		result *= autoExposure;
	} else {
		result *= (layer.data.iso / 100.0) * (1.0 / (layer.data.aperture * layer.data.aperture)) * layer.data.exposureTime;
	}

	// Darkness eye adaptation: blend toward a desaturated scotopic tint as adapted luminance
	// drops. isSkyPixel is a geometric fact (depth >= 1.0), not tied to which layer is active --
	// unlike the old isSky-indexed LayerData lookups elsewhere in this file, this one isn't a
	// special case that disappears with the layer generalization.
	vec3 scotopicTint = isSkyPixel ? vec3(0.15, 0.3, 0.6) : (result * vec3(0.70, 0.80, 0.90));
	result = ApplyPurkinjeShift(result, layer.data.adaptedLuminance, scotopicTint);

	// Accumulate Bloom
	float activeIntensity = params.intensity > 0.0 ? params.intensity : params.bloomIntensity;
	result += bloomColor * activeIntensity;

	// 3. White Balance
	float activeTemp = layer.data.whiteTemp != 0.0 ? (layer.data.whiteTemp - 6500.0) / 3000.0 : params.temperature;
	float activeTint = layer.data.whiteTint != 0.0 ? layer.data.whiteTint : params.tint;
	result *= calculateWhiteBalanceGain(activeTemp, activeTint);

	// 4. Contrast adjustment
	float contrastVal = params.contrast > 0.0 ? params.contrast : 1.0;
	if (contrastVal != 1.0) {
		result = max(vec3(0.0), (result - vec3(0.18)) * contrastVal + vec3(0.18));
	}

	// 5. ASC CDL Color Grading
	for (int i = 0; i < params.numCdlEntries; ++i) {
		CdlEntry entry = cdlEntries[i];
		if (entry.enabled == 0) continue;

		float cdlWeight = 1.0;
		if (entry.isMain == 0) {
			float dist = abs(linearZ - entry.targetDepth);
			float x = clamp(dist / max(entry.falloffWidth, 0.0001), 0.0, 1.0);
			cdlWeight = pow(1.0 - x, entry.falloffRate);
		}

		if (cdlWeight > 0.0) {
			vec3 graded = pow(max(result * entry.cdlSlope.rgb + entry.cdlOffset.rgb, 0.0), entry.cdlPower.rgb);
			float gradedLuma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
			graded = gradedLuma + entry.cdlSaturation * (graded - gradedLuma);
			result = mix(result, graded, cdlWeight);
		}
	}

	// 6. Tonemapping
	if (layer.data.toneMappingEnabled != 0) {
		int mode = layer.data.toneMapMode;
		if (mode == 5) { // Uchimura
			if (layer.data.autoTuneEnabled != 0) {
				result = uchimura(result, layer.data.autoUchimuraP, layer.data.autoUchimuraA, layer.data.autoUchimuraM, layer.data.autoUchimuraL, layer.data.autoUchimuraC, layer.data.autoUchimuraB);
			} else {
				result = uchimura(result, layer.data.uchimuraP, layer.data.uchimuraA, layer.data.uchimuraM, layer.data.uchimuraL, layer.data.uchimuraC, layer.data.uchimuraB);
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

	// 7. Gamma Correction
	float gammaVal = layer.data.gamma > 0.0 ? layer.data.gamma : 2.2;
	return pow(clamp(result, 0.0, 1.0), vec3(1.0 / gammaVal));
}

void main() {
	vec3 sceneColor = SAMPLE_NEAREST(params.hdrColorIndex, inUV).rgb;
	vec3 bloomColor = vec3(0.0);
	if (params.bloomEnabled != 0 && params.bloomBlurIndex > 0u) {
		bloomColor = SAMPLE_LINEAR(params.bloomBlurIndex, inUV).rgb;
	}

	float rawDepth = 0.0;
	if (params.depthTextureIndex > 0u) {
		rawDepth = SAMPLE_NEAREST(params.depthTextureIndex, inUV).r;
	}
	bool  isSkyPixel = rawDepth >= 1.0;
	float linearZ = linearizeDepth(rawDepth, uNearPlane, uFarPlane);

	// 1. Guided Upsampling for LTM. One shared low-res LTM pyramid feeds every layer
	// (bloom_downsample.comp blends each layer's contribution into it the same sequential way),
	// so the "guidance" exposure ratio used to compare against that fused result is blended
	// across layers here too, rather than pinned to a single hardcoded layer.
	vec3 ltmCorrectedColor = sceneColor;
	if (params.ltmFusedIndex > 0u && params.ltmExpMipIndex > 0u && params.ltmRes.x > 0.0 && params.ltmRes.y > 0.0) {
		float blendedAutoExp = 1.0;
		bool  anyLtmForGuidance = false;
		for (int i = 0; i < numExposureLayers; ++i) {
			ExposureLayer layer = layers[i];
			float weight = exposureLayerWeight(layer, linearZ, isSkyPixel);
			if (weight <= 0.0 || layer.data.ltmEnabled == 0) {
				continue;
			}
			float layerAutoExp = layer.data.targetLuminance / max(layer.data.avgLuma, 0.0001);
			blendedAutoExp = anyLtmForGuidance ? mix(blendedAutoExp, layerAutoExp, weight) : layerAutoExp;
			anyLtmForGuidance = true;
		}

		if (anyLtmForGuidance) {
			vec3 currentExposure = sceneColor * blendedAutoExp;
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

			ltmCorrectedColor *= finalMultiplier;
		}
	}

	// 2-7 (ApplyExposureLayerChain): run the full per-layer post chain -- exposure, Purkinje,
	// bloom accumulate, white balance, contrast, CDL grading, tonemap curve, gamma -- for every
	// layer whose depth range overlaps this pixel, blending results in array order. Same
	// sequential mix() idiom CdlGradingLayers' own entries already use, generalized from "color
	// grade only" to the whole post chain. The default single isMain layer reduces this to
	// exactly one iteration at weight 1.0, i.e. today's unchanged single-pass behavior.
	vec3 result = ltmCorrectedColor;
	bool anyLayerApplied = false;
	for (int i = 0; i < numExposureLayers; ++i) {
		ExposureLayer layer = layers[i];
		float weight = exposureLayerWeight(layer, linearZ, isSkyPixel);
		if (weight <= 0.0) {
			continue;
		}

		vec3 layerResult = ApplyExposureLayerChain(layer, ltmCorrectedColor, bloomColor, isSkyPixel, linearZ);
		result = anyLayerApplied ? mix(result, layerResult, weight) : layerResult;
		anyLayerApplied = true;
	}

	outColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
