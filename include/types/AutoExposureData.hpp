#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct LayerDataHost {
		float adaptedLuminance{0.3f};
		float targetLuminance{0.25f};
		float minExposure{0.01f};
		float maxExposure{25.0f};

		std::int32_t useAutoExposure{1};
		float        centerWeightTightness{4.0f};
		glm::vec2    focusPoint{0.5f, 0.5f};

		float histogramLowCutoff{0.1f};
		float histogramHighCutoff{0.95f};
		float speedUp{3.0f};
		float speedDown{1.0f};

		float minLuma{0.0f};
		float maxLuma{0.0f};
		float avgLuma{0.0f};
		float stdDevLuma{0.0f};

		float emaMinLuma{0.0f};
		float emaMaxLuma{0.0f};
		float emaAvgLuma{0.0f};
		float emaStdDevLuma{0.0f};

		std::int32_t autoTuneEnabled{1};
		float        minContrast{0.6f};
		float        maxContrast{1.3f};
		float        targetBrightness{1.0f};

		float autoUchimuraP{1.0f};
		float autoUchimuraA{1.0f};
		float autoUchimuraM{0.22f};
		float autoUchimuraL{0.4f};
		float autoUchimuraC{1.33f};
		float autoUchimuraB{0.0f};
		float exposureTime{0.008f}; // 1/125s default
		float iso{100.0f};

		float        uchimuraP{1.0f};
		float        uchimuraA{1.0f};
		float        uchimuraM{0.22f};
		float        uchimuraL{0.4f};
		float        uchimuraC{1.33f};
		float        uchimuraB{0.0f};
		std::int32_t toneMapMode{5};
		std::int32_t toneMappingEnabled{1};

		glm::vec4 cdlSlope{1.0f, 1.0f, 1.0f, 0.0f};
		glm::vec4 cdlOffset{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4 cdlPower{1.0f, 1.0f, 1.0f, 0.0f};
		float     cdlSaturation{1.0f};

		float whiteTemp{6500.0f};
		float whiteTint{0.0f};
		float aperture{8.0f};

		std::int32_t ltmEnabled{1};
		float        ltmEvSpread{2.0f};
		float        ltmTarget{0.5f};
		float        ltmSigma{0.2f};

		float ltmWeightContrast{0.0f};
		float ltmWeightSaturation{0.0f};
		float ltmWeightExposedness{1.0f};
		float ltmBoostLocalContrast{0.0f};

		float gamma{2.2f};
		// std430 rounds an array-of-structs element stride up to a multiple of 16 -- see
		// autoexposure.glsl's matching padding and the static_assert below.
		float _pad0{0.0f};
		float _pad1{0.0f};
		float _pad2{0.0f};

		std::uint32_t histogram[256]{0};
		// Snapshot of the previous frame's fully-accumulated histogram, written by
		// bloom_downsample.comp's update_layer_ae right before it zeroes `histogram` for this
		// frame's accumulation. `histogram` itself is always zero by the time s_exposureReadback's
		// frame-overlap-delayed copy (below) can observe it, since the reset is the last write to
		// that field before the buffer slot comes back around -- a settings UI reading live
		// histogram data must read this field instead.
		std::uint32_t histogramSnapshot[256]{0};
	};

	// bloom_downsample.comp indexes `layers[2]` as a std430 array of LayerData; std430 rounds an
	// array-of-structs element stride up to a multiple of 16 regardless of the struct's own
	// natural size, so if this ever stops being a multiple of 16, C++'s layers[1] and
	// workgroupCounter land at a different byte offset than the GPU actually uses (confirmed via
	// spirv-dis: adding a single unpadded float here once shifted workgroupCounter's real offset
	// to 2592 while C++ still computed 2568). Keep LayerDataHost's tail padded rather than
	// trusting the two languages' rounding to silently agree.
	static_assert(
		sizeof(LayerDataHost) % 16 == 0,
		"LayerDataHost size must be a multiple of 16 bytes -- see comment above"
	);

	// One depth-ranged auto-exposure/tonemap layer. Mirrors cdl_grading.glsl's CdlEntry/
	// CdlEntryHost pattern exactly (targetDepth/falloffWidth/falloffRate/priority/enabled/isMain)
	// -- std430 buffer layout, so field order/types here must match autoexposure.glsl's
	// ExposureLayer field-for-field. isMain entries always apply at full weight; every other
	// entry blends in based on how close its targetDepth is to the pixel's linearized depth,
	// falling off over falloffWidth at falloffRate -- same weight function CdlEntry already uses.
	//
	// isSky is a second hard override, alongside isMain: weight is 1.0 for a background pixel
	// (depth sentinel, nothing rendered -- the same `rawDepth >= 1.0` test the old hardcoded
	// Scene/Sky split used) and 0.0 otherwise, completely bypassing the depth-falloff math above.
	// targetDepth/falloffWidth/falloffRate don't apply to an isSky layer for the same reason they
	// don't apply to isMain -- the old split's weight was never a smooth function of depth, it was
	// a hard binary, and that's the behavior isSky restores. See s_exposureData's default below.
	struct ExposureLayerHost {
		LayerDataHost data{};
		float         targetDepth{0.0f};
		float         falloffWidth{1.0f};
		float         falloffRate{1.0f};
		std::int32_t  priority{0};
		std::int32_t  enabled{1};
		std::int32_t  isMain{0};
		std::int32_t  isSky{0};
		// A third hard override, same shape as isSky: weight 1.0 for a pixel the depth-stencil
		// system classified as cloud (GBufferDepth's stencil aspect == 2, written by
		// CloudCompositeNode), 0.0 otherwise -- bypassing the depth-falloff math exactly like
		// isMain/isSky. Lets a cloud pixel get its own dedicated layer instead of being bucketed
		// into isSky's stats just because it also fails the old binary depth>=1.0 test.
		std::int32_t  isCloud{0};
		// std430 rounds an array-of-structs element stride up to a multiple of 16 -- LayerDataHost
		// is already a multiple of 16 (its own static_assert above), and the 8 trailing scalars
		// here (32 bytes) already land on a 16-byte boundary, so no explicit padding field is
		// needed anymore (unlike before isCloud was added, when 7 scalars needed 4 bytes of padding
		// to reach one). See the static_assert below for what breaks if this drifts (same
		// spirv-dis-confirmed failure mode as LayerDataHost's own padding comment).
	};

	static_assert(
		sizeof(ExposureLayerHost) % 16 == 0,
		"ExposureLayerHost size must be a multiple of 16 bytes -- see comment above"
	);

	constexpr std::size_t kMaxExposureLayers = 8;

	struct ExposureDataHost {
		std::array<ExposureLayerHost, kMaxExposureLayers> layers{};
		std::int32_t                                      numLayers{1};
		std::uint32_t                                      workgroupCounter{0};

		// layers[0] is always the isMain entry (see s_exposureData below); depth-ranged layers
		// only ever live in layers[1..numLayers-1] -- same convention and the same AddEntry/
		// RemoveEntry shape as CdlGradingLayersHost (CdlGradingData.hpp).
		bool AddLayer() {
			if (static_cast<std::size_t>(numLayers) >= kMaxExposureLayers) {
				return false;
			}
			layers[static_cast<std::size_t>(numLayers)] = ExposureLayerHost{};
			++numLayers;
			return true;
		}

		void RemoveLayer(std::size_t index) {
			if (index == 0 || index >= static_cast<std::size_t>(numLayers)) {
				return;
			}
			for (std::size_t i = index; i + 1 < static_cast<std::size_t>(numLayers); ++i) {
				layers[i] = layers[i + 1];
			}
			--numLayers;
		}
	};

	// Defaults to two layers: an isMain layer covering the whole screen (this is what makes a
	// cloud in front of dim terrain no longer inherit the terrain's dim exposure -- a cloud
	// against open sky resolves through the isSky layer below instead, not this one), plus an
	// isSky layer restoring the old hardcoded Sky layer's own dedicated exposure/tonemap/white-
	// balance tuning for background pixels -- confirmed by real-device visual check that dropping
	// the Sky split entirely (the original single-layer default) lost something real: the sky
	// itself inheriting the scene's exposure looked wrong, even though the motivating cloud/
	// terrain bug was fixed. targetLuminance=0.5 (vs. LayerDataHost's own 0.25 default) matches
	// the sky's old hardcoded brightness target exactly (TonemapComputeNode::Init used to set this
	// by hand every Init() call; now it's just this layer's own default, editable at runtime like
	// any other). Both layers stay fully removable/editable via RemoveLayer()/the Lighting
	// widget -- this is a default, not a hardcoded special case.
	inline ExposureDataHost s_exposureData{
		.layers = {
			ExposureLayerHost{.isMain = 1},
			ExposureLayerHost{.data = {.targetLuminance = 0.5f}, .isSky = 1},
		},
		.numLayers = 2,
	};

	// Copies only the fields QuickSettingsWidget's ImGui sliders expose (see LayerDataHost's
	// field groups above) from src (s_exposureData, host-side) into dst (the mapped GPU buffer),
	// deliberately skipping every field bloom_downsample.comp's update_layer_ae owns: statistics
	// (minLuma/maxLuma/avgLuma/stdDevLuma), the EMAs, autoUchimura*, and histogram. Called every
	// frame so slider changes take effect without resetting the shader's own accumulated
	// auto-exposure/auto-tune state.
	inline void SyncAutoExposureTunables(LayerDataHost& dst, const LayerDataHost& src) {
		dst.targetLuminance = src.targetLuminance;
		dst.minExposure = src.minExposure;
		dst.maxExposure = src.maxExposure;
		dst.useAutoExposure = src.useAutoExposure;
		dst.exposureTime = src.exposureTime;
		dst.iso = src.iso;
		dst.aperture = src.aperture;
		dst.centerWeightTightness = src.centerWeightTightness;
		dst.focusPoint = src.focusPoint;
		dst.histogramLowCutoff = src.histogramLowCutoff;
		dst.histogramHighCutoff = src.histogramHighCutoff;
		dst.speedUp = src.speedUp;
		dst.speedDown = src.speedDown;

		dst.autoTuneEnabled = src.autoTuneEnabled;
		dst.minContrast = src.minContrast;
		dst.maxContrast = src.maxContrast;
		dst.targetBrightness = src.targetBrightness;

		dst.uchimuraP = src.uchimuraP;
		dst.uchimuraA = src.uchimuraA;
		dst.uchimuraM = src.uchimuraM;
		dst.uchimuraL = src.uchimuraL;
		dst.uchimuraC = src.uchimuraC;
		dst.uchimuraB = src.uchimuraB;
		dst.toneMapMode = src.toneMapMode;
		dst.toneMappingEnabled = src.toneMappingEnabled;

		dst.cdlSlope = src.cdlSlope;
		dst.cdlOffset = src.cdlOffset;
		dst.cdlPower = src.cdlPower;
		dst.cdlSaturation = src.cdlSaturation;

		dst.whiteTemp = src.whiteTemp;
		dst.whiteTint = src.whiteTint;

		dst.ltmEnabled = src.ltmEnabled;
		dst.ltmEvSpread = src.ltmEvSpread;
		dst.ltmTarget = src.ltmTarget;
		dst.ltmSigma = src.ltmSigma;
		dst.ltmWeightContrast = src.ltmWeightContrast;
		dst.ltmWeightSaturation = src.ltmWeightSaturation;
		dst.ltmWeightExposedness = src.ltmWeightExposedness;
		dst.ltmBoostLocalContrast = src.ltmBoostLocalContrast;

		dst.gamma = src.gamma;
	}

	// Snapshot of the GPU-owned fields (statistics, EMAs, autoUchimura*, histogram) that
	// bloom_downsample.comp's update_layer_ae writes -- s_exposureData never reflects these since
	// SyncAutoExposureTunables deliberately skips them. Engine::DrawFrame populates this every
	// frame by copying autoExposureMapped[activeFrame]'s current contents just before overwriting
	// it with this frame's tunables: by construction that memory was last written by the compute
	// dispatch FRAME_OVERLAP frames ago, and frame-overlap fencing already guarantees that
	// dispatch has retired by the time its buffer slot comes back around for reuse, so this read
	// needs no extra synchronization of its own. A settings UI wanting live stats/histogram
	// display should read from here, not from s_exposureData.
	inline ExposureDataHost s_exposureReadback{};

} // namespace brassica
