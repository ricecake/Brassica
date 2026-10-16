#include "ui/QuickSettingsWidget.hpp"

#include <algorithm>
#include <cmath>

#include "ConfigManager.hpp"
#include "EngineConstants.hpp"
#include "imgui.h"
#include "lighting/LightManager.hpp"
#include "particle/IParticleManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainManager.hpp"
#include "types/CameraData.hpp"

namespace brassica::ui {

	QuickSettingsWidget::QuickSettingsWidget() {
		m_visible = true;

		if (ServiceLocator::Instance().Has<ConfigManager>()) {
			auto& cfg = *ServiceLocator::Instance().Get<ConfigManager>();
			m_timeOfDay = cfg.GetAppSetting<float>("quick_day_night_time", 12.0f);
			m_timePaused = cfg.GetAppSetting<bool>("quick_day_night_paused", false);
			m_useLocalTime = cfg.GetAppSetting<bool>("quick_use_local_time", false);
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
				auto        cam = ServiceLocator::Instance().Get<CameraData>();
				const char* camModes[] = {"Instant", "Accelerated", "FirstPerson"};
				int         currentCamMode = static_cast<int>(cam->mode);
				if (ImGui::Combo("Mode##Cam", &currentCamMode, camModes, IM_ARRAYSIZE(camModes))) {
					cam->mode = static_cast<CameraMode>(currentCamMode);
				}
				ImGui::TextDisabled("(Press '=' key to cycle mode)");

				constexpr float R = FAKE_PLANET_RADIUS;
				float latDeg = std::clamp((cam->position.z / R) * (180.0f / glm::pi<float>()), -90.0f, 90.0f);
				float lonDeg = (cam->position.x / R) * (180.0f / glm::pi<float>());
				while (lonDeg > 180.0f) lonDeg -= 360.0f;
				while (lonDeg < -180.0f) lonDeg += 360.0f;
				float altM = cam->position.y;

				bool changed = false;
				changed |= ImGui::DragFloat("Latitude (deg)##Cam", &latDeg, 0.1f, -90.0f, 90.0f, "%.2f deg");
				changed |= ImGui::DragFloat("Longitude (deg)##Cam", &lonDeg, 0.1f, -180.0f, 180.0f, "%.2f deg");
				changed |= ImGui::DragFloat("Altitude (m)##Cam", &altM, 1.0f, -1000.0f, 1000000.0f, "%.1f m");

				if (changed) {
					latDeg = std::clamp(latDeg, -90.0f, 90.0f);
					while (lonDeg > 180.0f) lonDeg -= 360.0f;
					while (lonDeg < -180.0f) lonDeg += 360.0f;

					cam->position.z = (latDeg / (180.0f / glm::pi<float>())) * R;
					cam->position.x = (lonDeg / (180.0f / glm::pi<float>())) * R;
					cam->position.y = altM;
				}

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

			float camLongitudeDeg = 0.0f;
			if (ServiceLocator::Instance().Has<CameraData>()) {
				auto cam = ServiceLocator::Instance().Get<CameraData>();
				constexpr float R = FAKE_PLANET_RADIUS;
				camLongitudeDeg = (cam->position.x / R) * (180.0f / glm::pi<float>());
				while (camLongitudeDeg > 180.0f) camLongitudeDeg -= 360.0f;
				while (camLongitudeDeg < -180.0f) camLongitudeDeg += 360.0f;
			}

			if (ServiceLocator::Instance().Has<LightManager>()) {
				auto  lightMgr = ServiceLocator::Instance().Get<LightManager>();
				auto& cycle = lightMgr->GetDayNightCycle();
				m_timeOfDay = cycle.time;
				m_timePaused = cycle.paused;
			}

			if (ImGui::Checkbox("Use Local Time##Quick", &m_useLocalTime)) {
				if (cfg) {
					cfg->SetAppSetting("quick_use_local_time", m_useLocalTime);
				}
			}

			float displayTime = m_timeOfDay;
			if (m_useLocalTime) {
				displayTime = m_timeOfDay + (camLongitudeDeg / 15.0f);
				while (displayTime >= 24.0f) displayTime -= 24.0f;
				while (displayTime < 0.0f) displayTime += 24.0f;
			}

			const char* timeLabel = m_useLocalTime ? "Time (Local 24h)##Quick" : "Time (UTC 24h)##Quick";
			if (ImGui::SliderFloat(timeLabel, &displayTime, 0.0f, 24.0f, "%.1f h")) {
				if (m_useLocalTime) {
					m_timeOfDay = displayTime - (camLongitudeDeg / 15.0f);
					while (m_timeOfDay >= 24.0f) m_timeOfDay -= 24.0f;
					while (m_timeOfDay < 0.0f) m_timeOfDay += 24.0f;
				} else {
					m_timeOfDay = displayTime;
				}

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

			if (ServiceLocator::Instance().Has<LightManager>()) {
				auto lightMgr = ServiceLocator::Instance().Get<LightManager>();
				bool gtaoEnabled = lightMgr->IsGTAOEnabled();
				if (ImGui::Checkbox("Enable GTAO##Quick", &gtaoEnabled)) {
					lightMgr->SetGTAOEnabled(gtaoEnabled);
					if (cfg) {
						cfg->SetAppSetting("quick_gtao_enabled", gtaoEnabled);
					}
				}
				if (gtaoEnabled) {
					float radius = lightMgr->GetGTAORadius();
					if (ImGui::SliderFloat("AO Radius##Quick", &radius, 0.1f, 5.0f, "%.2f m")) {
						lightMgr->SetGTAORadius(radius);
					}

					float intensity = lightMgr->GetGTAOIntensity();
					if (ImGui::SliderFloat("AO Intensity##Quick", &intensity, 0.0f, 3.0f)) {
						lightMgr->SetGTAOIntensity(intensity);
					}

					float directInfluence = lightMgr->GetGTAODirectInfluence();
					if (ImGui::SliderFloat("AO Direct Light Influence##Quick", &directInfluence, 0.0f, 1.0f)) {
						lightMgr->SetGTAODirectInfluence(directInfluence);
					}

					bool debugVisualize = lightMgr->IsGTAODebugVisualizeEnabled();
					if (ImGui::Checkbox("Debug Visualize AO##Quick", &debugVisualize)) {
						lightMgr->SetGTAODebugVisualizeEnabled(debugVisualize);
					}
				}
			}

			ImGui::Separator();

			// Particles
			ImGui::TextColored(ImVec4(0, 1, 1, 1), "Particles:");
			if (ServiceLocator::Instance().Has<IParticleManager>()) {
				auto particleMgr = ServiceLocator::Instance().Get<IParticleManager>();
				bool particlesEnabled = particleMgr->IsEnabled();
				if (ImGui::Checkbox("Enable Particles##Quick", &particlesEnabled)) {
					particleMgr->SetEnabled(particlesEnabled);
					if (cfg) {
						cfg->SetAppSetting("particles_enabled", particlesEnabled);
					}
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
			if (ImGui::Button("Regenerate Terrain##Quick")) {
				if (ServiceLocator::Instance().Has<ITerrainManager>()) {
					auto terrainManager = ServiceLocator::Instance().Get<ITerrainManager>();
					terrainManager->Regenerate();
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
