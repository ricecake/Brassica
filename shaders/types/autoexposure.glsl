#ifndef TYPES_AUTOEXPOSURE_GLSL
#define TYPES_AUTOEXPOSURE_GLSL

// Auto-exposure layer data for post-processing, tone mapping, and grading. Tunables only --
// bloom_downsample.comp's update_layer_ae()-owned accumulator state (adaptedLuminance, stats,
// EMAs, auto-tuned Uchimura params, histogram) lives in the separate LayerState struct /
// AutoExposureStateBuffer below instead. They used to live here, in AutoExposureBuffer, which
// is allocated per-FRAME_OVERLAP-slot (a different real buffer per slot) -- fine for tunables
// re-synced from the CPU every frame, but wrong for continuously-accumulated GPU state: with
// FRAME_OVERLAP=2, that state only ever got touched every *other* frame, so there were
// effectively two independent, half-frame-rate-sampled adaptedLuminance sequences interleaving
// on screen every displayed frame -- visible as flashing/alternating brightness, worst right at
// startup (before either sequence converges) and during fast swings (lightning) where the two
// sequences sample the changing signal differently. See AutoExposureStateBuffer's own comment
// and Engine.cpp's matching creation-site comment for the fix.
struct LayerData {
	float targetLuminance;
	float minExposure;
	float maxExposure;

	int   useAutoExposure;
	float centerWeightTightness;
	vec2  focusPoint;

	float histogramLowCutoff;
	float histogramHighCutoff;
	float speedUp;
	float speedDown;

	// Auto-tune settings
	int   autoTuneEnabled;
	float minContrast;
	float maxContrast;
	float targetBrightness;

	float exposureTime;
	float iso;

	// Manual Uchimura parameters
	float uchimuraP;
	float uchimuraA;
	float uchimuraM;
	float uchimuraL;
	float uchimuraC;
	float uchimuraB;
	int   toneMapMode;
	int   toneMappingEnabled;

	// ASC CDL (vec4 for std430 alignment)
	vec4  cdlSlope;
	vec4  cdlOffset;
	vec4  cdlPower;
	float cdlSaturation;

	// White Balance
	float whiteTemp;
	float whiteTint;
	float aperture;

	// Local Tone Mapping (Exposure Fusion)
	int   ltmEnabled;
	float ltmEvSpread;
	float ltmTarget;
	float ltmSigma;

	float ltmWeightContrast;
	float ltmWeightSaturation;
	float ltmWeightExposedness;
	float ltmBoostLocalContrast;

	float gamma;
	// std430 rounds an array-of-structs element stride up to a multiple of 16 -- padding the
	// struct itself out to that boundary keeps this an explicit, checkable size instead of
	// relying on the two languages' rounding rules to silently agree. See LayerDataHost's
	// static_assert (AutoExposureData.hpp) for what breaks if this drifts.
	float _pad0;
	float _pad1;
};

// GPU-owned per-layer accumulator state that bloom_downsample.comp's update_layer_ae() reads
// and writes every frame -- see LayerData's comment above for why this can't live in the
// per-FRAME_OVERLAP-slot AutoExposureBuffer alongside the tunables. Mirrors LayerStateHost
// (AutoExposureData.hpp) field-for-field.
struct LayerState {
	float adaptedLuminance;

	// Statistics (recomputed fresh from this frame's histogram, no history of their own)
	float minLuma;
	float maxLuma;
	float avgLuma;
	float stdDevLuma;

	// EMAs
	float emaMinLuma;
	float emaMaxLuma;
	float emaAvgLuma;
	float emaStdDevLuma;

	// Auto-calculated Uchimura parameters
	float autoUchimuraP;
	float autoUchimuraA;
	float autoUchimuraM;
	float autoUchimuraL;
	float autoUchimuraC;
	float autoUchimuraB;

	// std430 rounds an array-of-structs element stride up to a multiple of 16 -- see
	// LayerStateHost's static_assert (AutoExposureData.hpp) for what breaks if this drifts.
	float _pad0;

	uint  histogram[256];
	// Snapshot of the previous frame's fully-accumulated histogram, written by
	// bloom_downsample.comp's update_layer_ae right before it zeroes `histogram` for this frame's
	// accumulation. CPU readback must read this field, not `histogram` -- by the time any readback
	// can see this buffer, `histogram` has already been reset to 0 (see AutoExposureData.hpp's
	// histogramSnapshot comment).
	uint  histogramSnapshot[256];
};

// One depth-ranged auto-exposure/tonemap layer. Mirrors cdl_grading.glsl's CdlEntry pattern
// exactly -- isMain layers always apply at full weight; every other layer blends in based on how
// close its targetDepth is to the pixel's linearized depth, falling off over falloffWidth at
// falloffRate. isSky is a second hard override: weight 1.0 for a background pixel (depth
// sentinel), 0.0 otherwise, bypassing the depth-falloff math entirely -- see AutoExposureData.hpp's
// ExposureLayerHost for the full explanation and the host-side mirror.
struct ExposureLayer {
	LayerData data;
	float     targetDepth;
	float     falloffWidth;
	float     falloffRate;
	int       priority;
	int       enabled;
	int       isMain;
	int       isSky;
	// A third hard override, same shape as isSky -- see AutoExposureData.hpp's ExposureLayerHost
	// comment for the full explanation (weight 1.0 for a stencil-classified cloud pixel).
	int       isCloud;
	// std430 rounds an array-of-structs element stride up to a multiple of 16 -- the 8 trailing
	// scalars above (targetDepth..isCloud) already land on a 16-byte boundary now, so no explicit
	// padding field is needed -- see AutoExposureData.hpp's matching comment and static_assert.
};

#define MAX_EXPOSURE_LAYERS 8

layout(std430, set = 0, binding = 5) buffer AutoExposureBuffer {
	ExposureLayer layers[MAX_EXPOSURE_LAYERS];
	int           numExposureLayers;
	// Cross-workgroup "last one out" signal for bloom_downsample.comp's single dispatch -- purely
	// transient within that one dispatch (reset to 0 by its own last workgroup every time), so
	// unlike LayerState below it's fine to leave per-FRAME_OVERLAP-slot.
	uint          workgroupCounter;
};

// One persistent buffer (not per-FRAME_OVERLAP-slot -- see Engine.cpp's AutoExposureStateBuffer
// creation comment, same single-allocation pattern as LightningStrikePool/ClusteredLighting),
// indexed the same way as AutoExposureBuffer's layers[] (same index = same logical layer).
#ifdef AUTOEXPOSURE_STATE_WRITABLE
layout(std430, set = 0, binding = 8) buffer AutoExposureStateBuffer {
	LayerState states[MAX_EXPOSURE_LAYERS];
};
#else
layout(std430, set = 0, binding = 8) readonly buffer AutoExposureStateBuffer {
	LayerState states[MAX_EXPOSURE_LAYERS];
};
#endif

// Same weight function cdl_grading.glsl's depth-ranged CdlEntry loop already uses. Callers loop
// i in 0..numExposureLayers and skip any layer with weight <= 0.0 (an isMain layer with
// enabled == 0 still blocks nothing else, so disabled layers simply contribute zero weight here
// rather than being filtered out of the loop itself). isSkyPixel is the same `depth >= 1.0`
// background test every caller already computes locally -- passed in rather than re-derived here
// since this function only ever sees linearZ, which saturates at the far plane and isn't a
// reliable binary signal on its own (unlike isMain, isSky is a hard override, not a smooth
// falloff, so it needs the real sentinel test, not a depth comparison).
//
// cloudWeight is NOT a classification bool like isSkyPixel -- it's the real local cloud opacity
// (1.0 - the cloud raymarch's own transmittance at this pixel, sampled directly from
// CloudAccumulatedTransmittance by every caller), continuous in [0, 1]. A hard binary cloud/not-
// cloud split (the first version of this) snapped a translucent edge pixel straight to 100% Cloud-
// layer treatment the instant any haze was detected, producing a visible seam -- an opaque cloud
// core still gets the Cloud layer at full weight, but a thin edge gets a small share of it and
// lets the normal back-to-front remainingWeight mechanism hand the rest to whatever's actually
// behind it (Sky or Scene), exactly like Sky itself falls back to Main.
float exposureLayerWeight(ExposureLayer layer, float linearZ, bool isSkyPixel, float cloudWeight) {
	if (layer.enabled == 0) {
		return 0.0;
	}
	if (layer.isMain != 0) {
		return 1.0;
	}
	if (layer.isCloud != 0) {
		return cloudWeight;
	}
	if (layer.isSky != 0) {
		return isSkyPixel ? 1.0 : 0.0;
	}
	if (isSkyPixel) {
		return 0.0;
	}
	float dist = abs(linearZ - layer.targetDepth);
	float x = clamp(dist / max(layer.falloffWidth, 0.0001), 0.0, 1.0);
	return pow(1.0 - x, layer.falloffRate);
}

#endif // TYPES_AUTOEXPOSURE_GLSL
