#include "ui/QuickSettingsWidget.hpp"

#include "ConfigManager.hpp"
#include "imgui.h"
#include "lighting/LightManager.hpp"
#include "ServiceLocator.hpp"

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
		ImGui::SetNextWindowSize(ImVec2(350, 300), ImGuiCond_FirstUseEver);

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

			// Mood Engine
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Mood Engine:");
			if (ImGui::Checkbox("Enable Mood Engine##Quick", &m_enableMood)) {
				if (cfg) {
					cfg->SetAppSetting("quick_mood_enabled", m_enableMood);
				}
			}

			ImGui::Separator();

			// Terrain
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Terrain:");
			if (ImGui::Checkbox("Render Terrain##Quick", &m_renderTerrain)) {
				if (cfg) {
					cfg->SetAppSetting("render_terrain", m_renderTerrain);
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
