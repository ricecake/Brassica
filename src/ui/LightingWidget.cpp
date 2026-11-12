#include "ui/LightingWidget.hpp"

#include <algorithm>
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

				// TEMPORARY debug aid -- see TonemapPushConstants.hpp's debugVisualizeStencil
				// comment. Replaces the final image with a flat magenta(cloud)/cyan(sky)/
				// black(scene) tint so the depth-stencil cloud classification can be checked in
				// isolation from exposure math. Remove once confirmed working on real hardware.
				bool debugVisualizeStencil = s_tonemapPush.debugVisualizeStencil != 0;
				if (ImGui::Checkbox("DEBUG: Visualize Cloud/Sky/Scene Stencil", &debugVisualizeStencil)) {
					s_tonemapPush.debugVisualizeStencil = debugVisualizeStencil ? 1 : 0;
				}

				if (bloomEnabled) {
					ImGui::SliderFloat("Intensity##Bloom", &s_tonemapPush.intensity, 0.0f, 2.0f);
					ImGui::SliderFloat("Threshold", &s_bloomDownsamplePush.threshold, 0.0f, 3.0f);
					ImGui::SliderFloat("Soft Knee", &s_bloomDownsamplePush.softness, 0.0f, 1.0f);
					ImGui::SliderFloat("Karis Dampening", &s_bloomDownsamplePush.karisDampening, 0.0f, 1.0f);
					ImGui::SliderFloat("Max Pooling Factor", &s_bloomDownsamplePush.maxPoolingFactor, 0.0f, 1.0f);
					ImGui::SliderFloat("Min Intensity (AE)", &s_tonemapPush.minIntensity, 0.0f, 5.0f);
					ImGui::SliderFloat("Max Intensity (AE)", &s_tonemapPush.maxIntensity, 0.0f, 10.0f);
				}

				// Exposure layers (AutoExposureData.hpp's ExposureDataHost) -- a dynamic list, not a
				// fixed Scene/Sky tab pair, since there's no longer a hardcoded isSky split: layer 0
				// is always the isMain entry (always full weight, no depth fade, not removable),
				// exactly like CdlGradingLayers' own entries[0] below. Uses the same
				// TreeNode-per-entry + Add/Remove pattern as the Color Grading section further
				// down, rather than ImGui tabs, since tabs don't handle dynamic add/remove as
				// cleanly as a plain list.
				auto drawExposureLayer = [&](int layerIdx, ExposureLayerHost& layerHost) {
					LayerDataHost& layer = layerHost.data;

					ImGui::Text("Auto-Exposure");
					bool aeEnabled = layer.useAutoExposure != 0;
					if (ImGui::Checkbox("Enable AE", &aeEnabled)) {
						layer.useAutoExposure = aeEnabled ? 1 : 0;
					}

					if (layer.useAutoExposure != 0) {
						ImGui::SliderFloat(
							"Screen Brightness (Key Value)",
							&layer.targetLuminance,
							0.01f,
							1.0f
						);
						ImGui::SliderFloat("Speed Up", &layer.speedUp, 0.1f, 10.0f);
						ImGui::SliderFloat("Speed Down", &layer.speedDown, 0.1f, 10.0f);
						ImGui::SliderFloat("Min EV100", &m_minEV100[layerIdx], -20.0f, 20.0f, "%.1f");
						ImGui::SliderFloat("Max EV100", &m_maxEV100[layerIdx], -20.0f, 20.0f, "%.1f");
						// EV100 -> exposure, recomputed every frame so it stays in sync with
						// targetLuminance regardless of which slider last changed (matches
						// the boidish source's BloomEffect::Apply, which recalculates this
						// every frame rather than only on edit).
						layer.minExposure = layer.targetLuminance /
							(1.2f * std::pow(2.0f, m_maxEV100[layerIdx]));
						layer.maxExposure = layer.targetLuminance /
							(1.2f * std::pow(2.0f, m_minEV100[layerIdx]));
						ImGui::SliderFloat("Center Weight", &layer.centerWeightTightness, 0.0f, 10.0f);
						ImGui::SliderFloat2("Focus Point", &layer.focusPoint.x, 0.0f, 1.0f);
						ImGui::SliderFloat("Histogram Low Cutoff", &layer.histogramLowCutoff, 0.0f, 1.0f);
						ImGui::SliderFloat("Histogram High Cutoff", &layer.histogramHighCutoff, 0.0f, 1.0f);

						ImGui::Separator();
						ImGui::Text("Statistics (Live Data):");
						const LayerDataHost& liveLayer = s_exposureReadback.layers[layerIdx].data;
						ImGui::Text(
							"Adapted Luma: %.4f | Avg: %.4f",
							liveLayer.adaptedLuminance,
							liveLayer.avgLuma
						);
						ImGui::Text(
							"Min Luma: %.4f | Max: %.4f | StdDev: %.4f",
							liveLayer.minLuma,
							liveLayer.maxLuma,
							liveLayer.stdDevLuma
						);

						if (ImGui::TreeNode("Luminance Histogram (256 bins)")) {
							std::vector<float> histValues(256);
							float              maxHistVal = 1.0f;
							for (int i = 0; i < 256; i++) {
								histValues[i] = static_cast<float>(liveLayer.histogramSnapshot[i]);
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
							if (ImGui::IsItemHovered()) {
								// Matches bloom_downsample.comp's MIN_LOG_LUMA/MAX_LOG_LUMA bin mapping exactly.
								constexpr float kMinLogLuma = -10.0f;
								constexpr float kMaxLogLuma = 14.0f;
								const ImVec2    rectMin = ImGui::GetItemRectMin();
								const ImVec2    rectMax = ImGui::GetItemRectMax();
								const float     relX =
									(ImGui::GetMousePos().x - rectMin.x) / std::max(rectMax.x - rectMin.x, 1.0f);
								const int bin = std::clamp(static_cast<int>(relX * 256.0f), 0, 255);
								const float binLogLuma =
									kMinLogLuma + (static_cast<float>(bin) + 0.5f) / 256.0f * (kMaxLogLuma - kMinLogLuma);
								ImGui::SetTooltip(
									"Bin %d | Luma: %.4f | Weight: %.0f",
									bin,
									std::exp(binLogLuma),
									histValues[bin]
								);
							}
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
						const char* modes[] = {
							"ACES",
							"Filmic",
							"Lottes",
							"Reinhard",
							"Reinhard II",
							"Uchimura",
							"Uncharted 2",
							"Unreal 3",
							"Debug"
						};
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
						ImGui::Text("White Balance");
						ImGui::SliderFloat("Temperature (K)", &layer.whiteTemp, 2000.0f, 12000.0f);
						ImGui::SliderFloat("Tint", &layer.whiteTint, -1.0f, 1.0f);

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
						ImGui::TreePop();
					}
				};

				if (ImGui::TreeNode("Main Layer")) {
					drawExposureLayer(0, s_exposureData.layers[0]);
					ImGui::TreePop();
				}

				ImGui::Separator();
				ImGui::Text("Depth-Ranged Exposure Layers");
				if (ImGui::Button("Add Layer##Exposure")) {
					s_exposureData.AddLayer();
				}

				for (int i = 1; i < s_exposureData.numLayers; ++i) {
					ImGui::PushID(i);
					ExposureLayerHost& layerHostForHeader = s_exposureData.layers[static_cast<std::size_t>(i)];
					std::string        headerName = layerHostForHeader.isSky != 0
						? "Layer " + std::to_string(i) + " (Sky)"
						: (layerHostForHeader.isCloud != 0 ? "Layer " + std::to_string(i) + " (Cloud)"
						                                   : "Layer " + std::to_string(i));
					bool removeRequested = false;
					if (ImGui::TreeNode(headerName.c_str())) {
						ExposureLayerHost& layerHost = s_exposureData.layers[static_cast<std::size_t>(i)];
						bool                layerEnabled = layerHost.enabled != 0;
						if (ImGui::Checkbox("Enabled##ExposureLayer", &layerEnabled)) {
							layerHost.enabled = layerEnabled ? 1 : 0;
						}

						// Sky overrides the normal depth-falloff weighting entirely (weight 1.0 on
						// background pixels, 0.0 otherwise) -- restores the old hardcoded Scene/Sky
						// split's binary behavior for whichever layer opts into it, same as isMain
						// overrides it the other way (always full weight). The depth/falloff/
						// priority controls below don't apply while this is on.
						bool isSky = layerHost.isSky != 0;
						if (ImGui::Checkbox("Sky Layer (background only)##ExposureLayer", &isSky)) {
							layerHost.isSky = isSky ? 1 : 0;
							if (isSky) {
								layerHost.isCloud = 0;
							}
						}

						// Cloud overrides the same way Sky does, but on the depth-stencil system's
						// cloud classification (GBufferDepth's stencil aspect == 2, written by
						// CloudCompositeNode) instead of the background depth sentinel -- lets a
						// layer target "visible cloud" specifically rather than inheriting Sky's
						// stats just because a raymarched cloud doesn't write real opaque depth
						// either. Mutually exclusive with Sky in this UI (isCloud is checked first
						// in exposureLayerWeight if both were somehow set, so this isn't a hazard
						// either way, but there's no reason to let a layer ask to be both at once).
						bool isCloud = layerHost.isCloud != 0;
						if (ImGui::Checkbox("Cloud Layer##ExposureLayer", &isCloud)) {
							layerHost.isCloud = isCloud ? 1 : 0;
							if (isCloud) {
								layerHost.isSky = 0;
							}
						}

						if (layerHost.isSky == 0 && layerHost.isCloud == 0) {
							ImGui::SliderFloat("Target Depth##ExposureLayer", &layerHost.targetDepth, 0.1f, 1000.0f);
							ImGui::SliderFloat("Falloff Width##ExposureLayer", &layerHost.falloffWidth, 0.1f, 500.0f);
							ImGui::SliderFloat("Falloff Rate##ExposureLayer", &layerHost.falloffRate, 0.1f, 10.0f);
							ImGui::InputInt("Priority##ExposureLayer", &layerHost.priority);
						}

						drawExposureLayer(i, layerHost);

						removeRequested = ImGui::Button("Remove Layer##ExposureLayer");
						ImGui::TreePop();
					}
					ImGui::PopID();
					if (removeRequested) {
						s_exposureData.RemoveLayer(static_cast<std::size_t>(i));
						break;
					}
				}
			}

			if (ImGui::CollapsingHeader("Color Grading (ASC CDL)")) {
				// Applies uniformly to every pixel regardless of which exposure layer is active
				// (tonemap.frag runs this once per exposure layer, before that layer's own tonemap
				// curve) -- not tied to any one exposure layer the way it used to be scene-only.
				ImGui::Text("Main Grade");
				DrawCdlFields(s_cdlGradingLayers.entries[0]);

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
						ImGui::SliderFloat(
							"Target Depth##CdlLayer",
							&entry.targetDepth,
							0.1f,
							1000.0f
						);
						ImGui::SliderFloat(
							"Falloff Width##CdlLayer",
							&entry.falloffWidth,
							0.1f,
							500.0f
						);
						ImGui::SliderFloat(
							"Falloff Rate##CdlLayer",
							&entry.falloffRate,
							0.1f,
							10.0f
						);
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
		}
		ImGui::End();
	}

} // namespace brassica::ui
