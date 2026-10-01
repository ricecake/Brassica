#include "ui/LightingWidget.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "imgui.h"
#include "types/AutoExposureData.hpp"
#include "types/BloomPushConstants.hpp"
#include "types/CdlGradingData.hpp"
#include "types/TonemapPushConstants.hpp"

namespace brassica::ui {

	namespace {

		void DrawCdlFields(CdlEntryHost& entry) {
			ImGui::ColorEdit3("Slope", &entry.cdlSlope.x);
			ImGui::ColorEdit3("Offset", &entry.cdlOffset.x);
			ImGui::ColorEdit3("Power", &entry.cdlPower.x);
			ImGui::SliderFloat("Saturation", &entry.cdlSaturation, 0.0f, 2.0f);
		}

	} // namespace

	LightingWidget::LightingWidget() {
		m_visible = true;
	}

	void LightingWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(20, 330), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Lighting", &m_visible)) {
			if (ImGui::CollapsingHeader("Bloom Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
				bool bloomEnabled = s_tonemapPush.bloomEnabled != 0;
				if (ImGui::Checkbox("Enable Bloom Blur", &bloomEnabled)) {
					s_tonemapPush.bloomEnabled = bloomEnabled ? 1 : 0;
				}
				ImGui::Checkbox("Enable Bloom/Auto-Exposure Compute Pass", &s_tonemapComputePassEnabled);

				if (bloomEnabled) {
					ImGui::SliderFloat("Intensity##Bloom", &s_tonemapPush.intensity, 0.0f, 2.0f);
					ImGui::SliderFloat("Threshold", &s_bloomDownsamplePush.threshold, 0.0f, 3.0f);
					ImGui::SliderFloat("Soft Knee", &s_bloomDownsamplePush.softness, 0.0f, 1.0f);
					ImGui::SliderFloat("Karis Dampening", &s_bloomDownsamplePush.karisDampening, 0.0f, 1.0f);
					ImGui::SliderFloat("Max Pooling Factor", &s_bloomDownsamplePush.maxPoolingFactor, 0.0f, 1.0f);
					ImGui::SliderFloat("Min Intensity (AE)", &s_tonemapPush.minIntensity, 0.0f, 5.0f);
					ImGui::SliderFloat("Max Intensity (AE)", &s_tonemapPush.maxIntensity, 0.0f, 10.0f);
				}

				if (ImGui::BeginTabBar("BloomTabs")) {
					auto drawLayer = [&](const char* label, int layerIdx, LayerDataHost& layer, bool isScene) {
						if (ImGui::BeginTabItem(label)) {
							ImGui::Text("Auto-Exposure");
							bool aeEnabled = layer.useAutoExposure != 0;
							if (ImGui::Checkbox("Enable AE", &aeEnabled)) {
								layer.useAutoExposure = aeEnabled ? 1 : 0;
							}

							if (layer.useAutoExposure != 0) {
								ImGui::SliderFloat("Screen Brightness (Key Value)", &layer.targetLuminance, 0.01f, 1.0f);
								ImGui::SliderFloat("Speed Up", &layer.speedUp, 0.1f, 10.0f);
								ImGui::SliderFloat("Speed Down", &layer.speedDown, 0.1f, 10.0f);
								ImGui::SliderFloat("Min EV100", &m_minEV100[layerIdx], -20.0f, 20.0f, "%.1f");
								ImGui::SliderFloat("Max EV100", &m_maxEV100[layerIdx], -20.0f, 20.0f, "%.1f");
								// EV100 -> exposure, recomputed every frame so it stays in sync with
								// targetLuminance regardless of which slider last changed (matches
								// the boidish source's BloomEffect::Apply, which recalculates this
								// every frame rather than only on edit).
								layer.minExposure = layer.targetLuminance / (1.2f * std::pow(2.0f, m_maxEV100[layerIdx]));
								layer.maxExposure = layer.targetLuminance / (1.2f * std::pow(2.0f, m_minEV100[layerIdx]));
								ImGui::SliderFloat("Center Weight", &layer.centerWeightTightness, 0.0f, 10.0f);
								ImGui::SliderFloat2("Focus Point", &layer.focusPoint.x, 0.0f, 1.0f);
								ImGui::SliderFloat("Histogram Low Cutoff", &layer.histogramLowCutoff, 0.0f, 1.0f);
								ImGui::SliderFloat("Histogram High Cutoff", &layer.histogramHighCutoff, 0.0f, 1.0f);

								ImGui::Separator();
								ImGui::Text("Statistics (Live Data):");
								const LayerDataHost& liveLayer = s_exposureReadback.layers[layerIdx];
								ImGui::Text("Adapted Luma: %.4f | Avg: %.4f", liveLayer.adaptedLuminance, liveLayer.avgLuma);
								ImGui::Text(
									"Min Luma: %.4f | Max: %.4f | StdDev: %.4f",
									liveLayer.minLuma,
									liveLayer.maxLuma,
									liveLayer.stdDevLuma
								);

								if (ImGui::TreeNode("Luminance Histogram (256 bins)")) {
									std::vector<float> histValues(256);
									float               maxHistVal = 1.0f;
									for (int i = 0; i < 256; i++) {
										histValues[i] = static_cast<float>(liveLayer.histogram[i]);
										if (histValues[i] > maxHistVal) {
											maxHistVal = histValues[i];
										}
									}
									ImGui::PlotHistogram(
										"##LumaHistogram",
										histValues.data(),
										256,
										0,
										nullptr,
										0.0f,
										maxHistVal,
										ImVec2(300, 80)
									);
									ImGui::TreePop();
								}
							} else {
								ImGui::SliderFloat("Exposure Time (s)", &layer.exposureTime, 0.0001f, 1.0f, "%.4f");
								ImGui::SliderFloat("ISO", &layer.iso, 10.0f, 6400.0f, "%.0f");
								ImGui::SliderFloat("Aperture (f-stop)", &layer.aperture, 1.0f, 22.0f, "%.1f");
							}

							ImGui::Separator();
							bool tmEnabled = layer.toneMappingEnabled != 0;
							if (ImGui::Checkbox("Tone Mapping", &tmEnabled)) {
								layer.toneMappingEnabled = tmEnabled ? 1 : 0;
							}
							if (layer.toneMappingEnabled != 0) {
								const char* modes[] =
									{"ACES", "Filmic", "Lottes", "Reinhard", "Reinhard II", "Uchimura", "Uncharted 2", "Unreal 3", "Debug"};
								ImGui::Combo("Mode", &layer.toneMapMode, modes, IM_ARRAYSIZE(modes));

								if (layer.toneMapMode == 5) { // Uchimura
									ImGui::SliderFloat("Max Brightness (P)", &layer.uchimuraP, 0.1f, 10.0f);
									ImGui::SliderFloat("Contrast (a)", &layer.uchimuraA, 0.1f, 5.0f);
									ImGui::SliderFloat("Linear Start (m)", &layer.uchimuraM, 0.0f, 1.0f);
									ImGui::SliderFloat("Linear Length (l)", &layer.uchimuraL, 0.0f, 1.0f);
									ImGui::SliderFloat("Black (c)", &layer.uchimuraC, 1.0f, 5.0f);
									ImGui::SliderFloat("Pedestal (b)", &layer.uchimuraB, 0.0f, 1.0f);
								}
							}

							ImGui::SliderFloat("Gamma", &layer.gamma, 1.0f, 3.0f, "%.2f");

							if (ImGui::TreeNode("Color Pipeline")) {
								bool autoTune = layer.autoTuneEnabled != 0;
								if (ImGui::Checkbox("Enable Auto-tune", &autoTune)) {
									layer.autoTuneEnabled = autoTune ? 1 : 0;
								}
								if (layer.autoTuneEnabled != 0) {
									ImGui::SliderFloat("Min Contrast", &layer.minContrast, 0.1f, 1.0f);
									ImGui::SliderFloat("Max Contrast", &layer.maxContrast, 1.0f, 5.0f);
									ImGui::SliderFloat("Target Brightness", &layer.targetBrightness, 0.1f, 2.0f);
								}

								ImGui::Separator();
								ImGui::Text("ASC CDL");
								if (isScene) {
									// Scene grading flows entirely through CdlGradingLayers' isMain
									// entry (entries[0]) now, not layer.cdlSlope/Offset/Power --
									// tonemap.frag's depth-based grading loop for isSky == 0 never
									// reads the flat per-layer CDL fields at all.
									DrawCdlFields(s_cdlGradingLayers.entries[0]);
								} else {
									ImGui::ColorEdit3("Slope", &layer.cdlSlope.x);
									ImGui::ColorEdit3("Offset", &layer.cdlOffset.x);
									ImGui::ColorEdit3("Power", &layer.cdlPower.x);
									ImGui::SliderFloat("Saturation", &layer.cdlSaturation, 0.0f, 2.0f);
								}

								if (isScene) {
									ImGui::Separator();
									ImGui::Text("Depth-Based Grading Layers");
									if (ImGui::Button("Add Layer##CDL")) {
										s_cdlGradingLayers.AddEntry();
									}

									for (int i = 1; i < s_cdlGradingLayers.numEntries; ++i) {
										ImGui::PushID(i);
										std::string headerName = "Layer " + std::to_string(i);
										bool        removeRequested = false;
										if (ImGui::TreeNode(headerName.c_str())) {
											CdlEntryHost& entry = s_cdlGradingLayers.entries[i];
											bool          entryEnabled = entry.enabled != 0;
											if (ImGui::Checkbox("Enabled##CdlLayer", &entryEnabled)) {
												entry.enabled = entryEnabled ? 1 : 0;
											}
											ImGui::SliderFloat("Target Depth##CdlLayer", &entry.targetDepth, 0.1f, 1000.0f);
											ImGui::SliderFloat("Falloff Width##CdlLayer", &entry.falloffWidth, 0.1f, 500.0f);
											ImGui::SliderFloat("Falloff Rate##CdlLayer", &entry.falloffRate, 0.1f, 10.0f);
											ImGui::InputInt("Priority##CdlLayer", &entry.priority);

											DrawCdlFields(entry);

											removeRequested = ImGui::Button("Remove Layer##CdlLayer");
											ImGui::TreePop();
										}
										ImGui::PopID();
										if (removeRequested) {
											s_cdlGradingLayers.RemoveEntry(static_cast<std::size_t>(i));
											break;
										}
									}
								}

								ImGui::Separator();
								ImGui::Text("White Balance");
								ImGui::SliderFloat("Temperature (K)", &layer.whiteTemp, 2000.0f, 12000.0f);
								ImGui::SliderFloat("Tint", &layer.whiteTint, -1.0f, 1.0f);

								if (isScene) {
									ImGui::Separator();
									ImGui::Text("Local Tone Mapping (Exposure Fusion)");
									bool ltmEnabled = layer.ltmEnabled != 0;
									if (ImGui::Checkbox("Enable LTM", &ltmEnabled)) {
										layer.ltmEnabled = ltmEnabled ? 1 : 0;
									}
									if (layer.ltmEnabled != 0) {
										ImGui::SliderFloat("EV Spread", &layer.ltmEvSpread, 0.0f, 4.0f);
										ImGui::SliderFloat("Well Exposed Target", &layer.ltmTarget, 0.0f, 1.0f);
										ImGui::SliderFloat("Well Exposed Sigma", &layer.ltmSigma, 0.01f, 1.0f);
										ImGui::SliderFloat("Weight Contrast", &layer.ltmWeightContrast, 0.0f, 1.0f);
										ImGui::SliderFloat("Weight Saturation", &layer.ltmWeightSaturation, 0.0f, 1.0f);
										ImGui::SliderFloat("Weight Exposedness", &layer.ltmWeightExposedness, 0.0f, 1.0f);
										ImGui::SliderFloat("Boost Local Contrast", &layer.ltmBoostLocalContrast, 0.0f, 2.0f);
									}
								}
								ImGui::TreePop();
							}

							ImGui::EndTabItem();
						}
					};

					drawLayer("Scene", 0, s_exposureData.layers[0], true);
					drawLayer("Sky", 1, s_exposureData.layers[1], false);

					ImGui::EndTabBar();
				}
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
