#include "ui/QuickSettingsWidget.hpp"

#include <algorithm>
#include <vector>

#include "ConfigManager.hpp"
#include "imgui.h"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"
#include "types/AutoExposureData.hpp"
#include "terrain/ITerrainClipmap.hpp"
#include "types/CameraData.hpp"
#include "types/TonemapPushConstants.hpp"

namespace brassica::ui {

	QuickSettingsWidget::QuickSettingsWidget() {
		m_visible = true;

		if (ServiceLocator::Instance().Has<ConfigManager>()) {
			auto& cfg = *ServiceLocator::Instance().Get<ConfigManager>();
			m_timeOfDay = cfg.GetAppSetting<float>("quick_day_night_time", 12.0f);
			m_timePaused = cfg.GetAppSetting<bool>("quick_day_night_paused", false);
			m_enableMood = cfg.GetAppSetting<bool>("quick_mood_enabled", true);
			m_renderTerrain = cfg.GetAppSetting<bool>("render_terrain", true);
			m_enableVolumetricLighting = cfg.GetAppSetting<bool>("enable_volumetric_lighting", true);
			m_volumetricIntensity = cfg.GetAppSetting<float>("volumetric_intensity", 1.0f);
			m_enableAtmosphere = cfg.GetAppSetting<bool>("enable_atmosphere", true);
			m_cloudCoverage = cfg.GetAppSetting<float>("quick_cloud_coverage", 0.3f);
		}
	}

	void QuickSettingsWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 175.0f, 30.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(350, 450), ImGuiCond_FirstUseEver);

		ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Quick Controls", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			ConfigManager* cfg = ServiceLocator::Instance().Has<ConfigManager>()
				? ServiceLocator::Instance().Get<ConfigManager>().get()
				: nullptr;

			// Camera
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Camera:");
			if (ServiceLocator::Instance().Has<CameraData>()) {
				auto cam = ServiceLocator::Instance().Get<CameraData>();
				const char* camModes[] = {"Instant", "Accelerated"};
				int currentCamMode = static_cast<int>(cam->mode);
				if (ImGui::Combo("Mode##Cam", &currentCamMode, camModes, IM_ARRAYSIZE(camModes))) {
					cam->mode = static_cast<CameraMode>(currentCamMode);
				}
				ImGui::TextDisabled("(Press '=' key to cycle mode)");

				ImGui::SliderFloat("Max Speed##Cam", &cam->speed, cam->minSpeed, cam->maxSpeed, "%.1f m/s");

				float baseFovDeg = glm::degrees(cam->baseFov);
				if (ImGui::SliderFloat("Base FOV##Cam", &baseFovDeg, 30.0f, 120.0f, "%.1f deg")) {
					cam->baseFov = glm::radians(baseFovDeg);
					if (cam->mode == CameraMode::Instant) {
						cam->fov = cam->baseFov;
					}
				}

				if (cam->mode == CameraMode::Accelerated) {
					ImGui::Text("Current Speed: %.1f m/s", cam->currentSpeed);
				}
			}

			ImGui::Separator();

			// Day/Night cycle
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Day/Night Cycle:");
			if (ServiceLocator::Instance().Has<LightManager>()) {
				auto  lightMgr = ServiceLocator::Instance().Get<LightManager>();
				auto& cycle = lightMgr->GetDayNightCycle();
				m_timeOfDay = cycle.time;
				m_timePaused = cycle.paused;
			}

			if (ImGui::SliderFloat("Time (24h)##Quick", &m_timeOfDay, 0.0f, 24.0f, "%.1f h")) {
				if (ServiceLocator::Instance().Has<LightManager>()) {
					auto  lightMgr = ServiceLocator::Instance().Get<LightManager>();
					auto& cycle = lightMgr->GetDayNightCycle();
					cycle.time = m_timeOfDay;
				}
				if (cfg) {
					cfg->SetAppSetting("quick_day_night_time", m_timeOfDay);
				}
			}

			if (ImGui::Checkbox("Pause Day/Night Cycle##Quick", &m_timePaused)) {
				if (ServiceLocator::Instance().Has<LightManager>()) {
					auto  lightMgr = ServiceLocator::Instance().Get<LightManager>();
					auto& cycle = lightMgr->GetDayNightCycle();
					cycle.paused = m_timePaused;
				}
				if (cfg) {
					cfg->SetAppSetting("quick_day_night_paused", m_timePaused);
				}
			}

			ImGui::Separator();

			// Exposure & Post-Processing Tone / CDL / Auto-Exposure System
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Exposure & Tone Mapping:");
			ImGui::SliderFloat("Exposure##Tone", &s_tonemapPush.exposure, 0.01f, 10.0f, "%.2f");

			if (ImGui::BeginTabBar("AutoExposureTabs")) {
				auto drawLayerSettings = [&](const char* label, LayerDataHost& layer, bool isScene) {
					if (ImGui::BeginTabItem(label)) {
						bool aeEnabled = layer.useAutoExposure != 0;
						if (ImGui::Checkbox("Enable Auto-Exposure", &aeEnabled)) {
							layer.useAutoExposure = aeEnabled ? 1 : 0;
						}

						if (layer.useAutoExposure != 0) {
							ImGui::SliderFloat("Target Luminance", &layer.targetLuminance, 0.01f, 1.0f);
							ImGui::SliderFloat("Min Exposure", &layer.minExposure, 0.001f, 10.0f);
							ImGui::SliderFloat("Max Exposure", &layer.maxExposure, 1.0f, 100.0f);
							ImGui::SliderFloat("Speed Up (Adaptation)", &layer.speedUp, 0.1f, 20.0f);
							ImGui::SliderFloat("Speed Down (Adaptation)", &layer.speedDown, 0.1f, 20.0f);
							ImGui::SliderFloat("Center Weight Tightness", &layer.centerWeightTightness, 0.0f, 10.0f);
							ImGui::SliderFloat2("Focus Point", &layer.focusPoint.x, 0.0f, 1.0f);
							ImGui::SliderFloat("Histogram Low Cutoff", &layer.histogramLowCutoff, 0.0f, 1.0f);
							ImGui::SliderFloat("Histogram High Cutoff", &layer.histogramHighCutoff, 0.0f, 1.0f);

							ImGui::Separator();
							ImGui::Text("Statistics (Live Data):");
							ImGui::Text("Adapted Luma: %.4f | Avg: %.4f", layer.adaptedLuminance, layer.avgLuma);
							ImGui::Text("Min Luma: %.4f | Max: %.4f | StdDev: %.4f", layer.minLuma, layer.maxLuma, layer.stdDevLuma);

							if (ImGui::TreeNode("Luminance Histogram (256 bins)")) {
								std::vector<float> histValues(256);
								float maxHistVal = 1.0f;
								for (int i = 0; i < 256; i++) {
									histValues[i] = static_cast<float>(layer.histogram[i]);
									if (histValues[i] > maxHistVal) {
										maxHistVal = histValues[i];
									}
								}
								ImGui::PlotHistogram("##LumaHistogram", histValues.data(), 256, 0, nullptr, 0.0f, maxHistVal, ImVec2(300, 80));
								ImGui::TreePop();
							}
						}

						ImGui::Separator();
						bool tmEnabled = layer.toneMappingEnabled != 0;
						if (ImGui::Checkbox("Enable Tone Mapping", &tmEnabled)) {
							layer.toneMappingEnabled = tmEnabled ? 1 : 0;
						}

						if (layer.toneMappingEnabled != 0) {
							const char* modes[] = {"ACES", "Filmic", "Lottes", "Reinhard", "Reinhard II", "Uchimura", "Uncharted 2", "Unreal", "Debug"};
							ImGui::Combo("Mode##LayerTM", &layer.toneMapMode, modes, IM_ARRAYSIZE(modes));

							if (layer.toneMapMode == 5) { // Uchimura
								bool autoTune = layer.autoTuneEnabled != 0;
								if (ImGui::Checkbox("Enable Auto-Tune Uchimura", &autoTune)) {
									layer.autoTuneEnabled = autoTune ? 1 : 0;
								}
								if (layer.autoTuneEnabled != 0) {
									ImGui::SliderFloat("Min Contrast", &layer.minContrast, 0.1f, 1.0f);
									ImGui::SliderFloat("Max Contrast", &layer.maxContrast, 1.0f, 5.0f);
									ImGui::SliderFloat("Target Brightness", &layer.targetBrightness, 0.1f, 2.0f);
									ImGui::TextDisabled("Auto P: %.2f | A: %.2f | M: %.2f", layer.autoUchimuraP, layer.autoUchimuraA, layer.autoUchimuraM);
								} else {
									ImGui::SliderFloat("Max Brightness (P)", &layer.uchimuraP, 0.1f, 10.0f);
									ImGui::SliderFloat("Contrast (a)", &layer.uchimuraA, 0.1f, 5.0f);
									ImGui::SliderFloat("Linear Start (m)", &layer.uchimuraM, 0.0f, 1.0f);
									ImGui::SliderFloat("Linear Length (l)", &layer.uchimuraL, 0.0f, 1.0f);
									ImGui::SliderFloat("Black (c)", &layer.uchimuraC, 1.0f, 5.0f);
									ImGui::SliderFloat("Pedestal (b)", &layer.uchimuraB, 0.0f, 1.0f);
								}
							}
						}

						if (ImGui::TreeNode("ASC CDL & White Balance")) {
							ImGui::ColorEdit3("Slope##LayerCDL", &layer.cdlSlope.x);
							ImGui::ColorEdit3("Offset##LayerCDL", &layer.cdlOffset.x);
							ImGui::ColorEdit3("Power##LayerCDL", &layer.cdlPower.x);
							ImGui::SliderFloat("Saturation##LayerCDL", &layer.cdlSaturation, 0.0f, 3.0f);

							ImGui::Separator();
							ImGui::SliderFloat("Temperature (K)", &layer.whiteTemp, 2000.0f, 12000.0f);
							ImGui::SliderFloat("Tint", &layer.whiteTint, -1.0f, 1.0f);

							if (isScene) {
								ImGui::Separator();
								bool ltmEnabled = layer.ltmEnabled != 0;
								if (ImGui::Checkbox("Enable Local Tone Mapping (LTM)", &ltmEnabled)) {
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

				drawLayerSettings("Scene Layer", s_exposureData.layers[0], true);
				drawLayerSettings("Sky Layer", s_exposureData.layers[1], false);

				ImGui::EndTabBar();
			}

			const char* toneModes[] = {
				"ACES",
				"Filmic",
				"Lottes",
				"Reinhard",
				"Reinhard2",
				"Uchimura",
				"Uncharted 2",
				"Unreal",
				"Debug",
				"None"
			};
			int currentMode = static_cast<int>(s_tonemapPush.toneMapMode);
			if (ImGui::Combo("Fallback Mode##Tone", &currentMode, toneModes, IM_ARRAYSIZE(toneModes))) {
				s_tonemapPush.toneMapMode = static_cast<std::uint32_t>(currentMode);
			}

			ImGui::Separator();

			// Terrain
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Terrain:");
			if (ImGui::Checkbox("Render Terrain##Quick", &m_renderTerrain)) {
				if (cfg) {
					cfg->SetAppSetting("render_terrain", m_renderTerrain);
				}
			}
			if (ImGui::Button("Regenerate Terrain##Quick")) {
				if (ServiceLocator::Instance().Has<ITerrainClipmap>()) {
					auto terrainClipmap = ServiceLocator::Instance().Get<ITerrainClipmap>();
					terrainClipmap->Regenerate();
				}
			}

			ImGui::Separator();

			// Volumetric Lighting
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Volumetric Lighting:");
			if (ImGui::Checkbox("Enable Volumetric Lighting##Quick", &m_enableVolumetricLighting)) {
				if (cfg) {
					cfg->SetAppSetting("enable_volumetric_lighting", m_enableVolumetricLighting);
				}
			}
			if (m_enableVolumetricLighting) {
				if (ImGui::SliderFloat("Volumetric Intensity##Quick", &m_volumetricIntensity, 0.0f, 5.0f)) {
					if (cfg) {
						cfg->SetAppSetting("volumetric_intensity", m_volumetricIntensity);
					}
				}
			}

			ImGui::Separator();

			// Atmosphere
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Atmosphere:");
			if (ImGui::Checkbox("Enable Atmosphere##Quick", &m_enableAtmosphere)) {
				if (cfg) {
					cfg->SetAppSetting("enable_atmosphere", m_enableAtmosphere);
				}
			}
			if (m_enableAtmosphere) {
				if (ImGui::SliderFloat("Cloud Coverage##Quick", &m_cloudCoverage, 0.0f, 1.0f)) {
					if (cfg) {
						cfg->SetAppSetting("quick_cloud_coverage", m_cloudCoverage);
					}
				}
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
