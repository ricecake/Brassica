#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	struct LayerDataHost {
		float     adaptedLuminance{0.3f};
		float     targetLuminance{0.25f};
		float     minExposure{0.01f};
		float     maxExposure{25.0f};

		std::int32_t useAutoExposure{1};
		float     centerWeightTightness{4.0f};
		glm::vec2 focusPoint{0.5f, 0.5f};

		float     histogramLowCutoff{0.1f};
		float     histogramHighCutoff{0.95f};
		float     speedUp{3.0f};
		float     speedDown{1.0f};

		float     minLuma{0.0f};
		float     maxLuma{0.0f};
		float     avgLuma{0.0f};
		float     stdDevLuma{0.0f};

		float     emaMinLuma{0.0f};
		float     emaMaxLuma{0.0f};
		float     emaAvgLuma{0.0f};
		float     emaStdDevLuma{0.0f};

		std::int32_t autoTuneEnabled{1};
		float     minContrast{0.6f};
		float     maxContrast{1.3f};
		float     targetBrightness{1.0f};

		float     autoUchimuraP{1.0f};
		float     autoUchimuraA{1.0f};
		float     autoUchimuraM{0.22f};
		float     autoUchimuraL{0.4f};
		float     autoUchimuraC{1.33f};
		float     autoUchimuraB{0.0f};
		float     exposureTime{0.008f}; // 1/125s default
		float     iso{100.0f};

		float     uchimuraP{1.0f};
		float     uchimuraA{1.0f};
		float     uchimuraM{0.22f};
		float     uchimuraL{0.4f};
		float     uchimuraC{1.33f};
		float     uchimuraB{0.0f};
		std::int32_t toneMapMode{5};
		std::int32_t toneMappingEnabled{1};

		glm::vec4 cdlSlope{1.0f, 1.0f, 1.0f, 0.0f};
		glm::vec4 cdlOffset{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4 cdlPower{1.0f, 1.0f, 1.0f, 0.0f};
		float     cdlSaturation{1.0f};

		float     whiteTemp{6500.0f};
		float     whiteTint{0.0f};
		float     aperture{8.0f};

		std::int32_t ltmEnabled{1};
		float     ltmEvSpread{2.0f};
		float     ltmTarget{0.5f};
		float     ltmSigma{0.2f};

		float     ltmWeightContrast{0.0f};
		float     ltmWeightSaturation{0.0f};
		float     ltmWeightExposedness{1.0f};
		float     ltmBoostLocalContrast{0.0f};

		std::uint32_t histogram[256]{0};
	};

	struct ExposureDataHost {
		LayerDataHost layers[2];
		std::uint32_t workgroupCounter{0};
	};

	inline ExposureDataHost s_exposureData{};

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
	}

} // namespace brassica
