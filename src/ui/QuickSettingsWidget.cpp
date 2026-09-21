#include "ui/QuickSettingsWidget.hpp"

#include "ConfigManager.hpp"
#include "imgui.h"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainClipmap.hpp"
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

			// Exposure & Post-Processing Tone / CDL
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Exposure & Tone Mapping:");
			ImGui::SliderFloat("Exposure##Tone", &s_tonemapPush.exposure, 0.01f, 10.0f, "%.2f");

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
			if (ImGui::Combo("Mode##Tone", &currentMode, toneModes, IM_ARRAYSIZE(toneModes))) {
				s_tonemapPush.toneMapMode = static_cast<std::uint32_t>(currentMode);
			}

			if (currentMode == 5) { // Uchimura
				ImGui::SliderFloat("Uchimura P##Tone", &s_tonemapPush.pMax, 0.1f, 5.0f);
				ImGui::SliderFloat("Uchimura a##Tone", &s_tonemapPush.pA, 0.1f, 5.0f);
				ImGui::SliderFloat("Uchimura m##Tone", &s_tonemapPush.pM, 0.0f, 1.0f);
				ImGui::SliderFloat("Uchimura l##Tone", &s_tonemapPush.pL, 0.0f, 1.0f);
				ImGui::SliderFloat("Uchimura c##Tone", &s_tonemapPush.pC, 0.1f, 5.0f);
				ImGui::SliderFloat("Uchimura b##Tone", &s_tonemapPush.pB, 0.0f, 1.0f);
			}

			ImGui::SliderFloat("Contrast##Tone", &s_tonemapPush.contrast, 0.0f, 3.0f);
			ImGui::SliderFloat("Saturation##Tone", &s_tonemapPush.saturation, 0.0f, 3.0f);
			ImGui::SliderFloat("Temperature##Tone", &s_tonemapPush.temperature, -1.0f, 1.0f);
			ImGui::SliderFloat("Tint##Tone", &s_tonemapPush.tint, -1.0f, 1.0f);

			if (ImGui::TreeNode("ASC CDL Color Grading")) {
				ImGui::ColorEdit3("Slope##CDL", &s_tonemapPush.cdlSlope.x);
				ImGui::ColorEdit3("Offset##CDL", &s_tonemapPush.cdlOffset.x);
				ImGui::ColorEdit3("Power##CDL", &s_tonemapPush.cdlPower.x);
				ImGui::SliderFloat("CDL Saturation##CDL", &s_tonemapPush.cdlSaturation, 0.0f, 3.0f);
				ImGui::TreePop();
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
