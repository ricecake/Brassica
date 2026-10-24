#include "ui/AudioWidget.hpp"

#include "audio/IAudioManager.hpp"
#include "ConfigManager.hpp"
#include "imgui.h"
#include "ServiceLocator.hpp"

namespace brassica::ui {

	AudioWidget::AudioWidget() {
		m_visible = false;
	}

	void AudioWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(100.0f, 100.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(380.0f, 520.0f), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Audio Settings", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			if (!ServiceLocator::Instance().Has<IAudioManager>()) {
				ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "IAudioManager not available");
				ImGui::End();
				return;
			}

			auto mgr = ServiceLocator::Instance().Get<IAudioManager>();
			if (!mgr) {
				ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "IAudioManager instance is null");
				ImGui::End();
				return;
			}

			auto state = mgr->GetState();
			bool modified = false;

			if (ImGui::CollapsingHeader("Total Audio Output", ImGuiTreeNodeFlags_DefaultOpen)) {
				modified |= ImGui::SliderFloat("Master Volume", &state.masterVolume, 0.0f, 1.0f, "%.2f");
				modified |= ImGui::SliderFloat("Music Volume", &state.musicVolume, 0.0f, 1.0f, "%.2f");
				modified |= ImGui::SliderFloat("SFX Volume", &state.sfxVolume, 0.0f, 1.0f, "%.2f");
				modified |= ImGui::SliderFloat("Global Pitch", &state.globalPitch, 0.1f, 2.0f, "%.2f");
			}

			if (ImGui::CollapsingHeader("Wind Sound Effect", ImGuiTreeNodeFlags_DefaultOpen)) {
				modified |= ImGui::Checkbox("Enable Wind Effect", &state.enableWind);
				if (state.enableWind) {
					modified |= ImGui::SliderFloat("Wind Volume Level", &state.windVolume, 0.0f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Wind Pitch / Frequency", &state.windPitch, 0.1f, 3.0f, "%.2f");
					modified |= ImGui::SliderFloat("Wind Resonance", &state.windResonance, 0.1f, 3.0f, "%.2f");
				}
			}

			if (ImGui::CollapsingHeader("Rain Sound Effect", ImGuiTreeNodeFlags_DefaultOpen)) {
				modified |= ImGui::Checkbox("Enable Rain Effect", &state.enableRain);
				if (state.enableRain) {
					modified |= ImGui::SliderFloat("Rain Volume Level", &state.rainVolume, 0.0f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rain Pitch / Filter", &state.rainPitch, 0.1f, 3.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rain Drop Density", &state.rainDensity, 0.1f, 3.0f, "%.2f");
				}
			}

			if (ImGui::CollapsingHeader("Foliage Rustle Effect")) {
				modified |= ImGui::Checkbox("Enable Rustle Effect", &state.enableRustle);
				if (state.enableRustle) {
					modified |= ImGui::SliderFloat("Rustle Volume Level", &state.rustleVolume, 0.0f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rustle Pitch / Cutoff", &state.rustlePitch, 0.1f, 3.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rustle Modulation Speed", &state.rustleModSpeed, 0.1f, 3.0f, "%.2f");
				}
			}

			if (ImGui::CollapsingHeader("Space Noise Effect")) {
				modified |= ImGui::Checkbox("Enable Space Noise Effect", &state.enableSpaceNoise);
				if (state.enableSpaceNoise) {
					modified |= ImGui::SliderFloat("Space Volume Level", &state.spaceVolume, 0.0f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Space Pitch / Cutoff", &state.spacePitch, 0.1f, 3.0f, "%.2f");
					modified |= ImGui::SliderFloat("Space Sub-Bass Freq (Hz)", &state.spaceSubFreq, 10.0f, 120.0f, "%.1f Hz");
				}
			}

			if (ImGui::CollapsingHeader("Speed Woosh Effect")) {
				modified |= ImGui::Checkbox("Enable Speed Woosh Effect", &state.enableSpeedWoosh);
				if (state.enableSpeedWoosh) {
					modified |= ImGui::SliderFloat("Speed Woosh Volume Level", &state.speedWooshVolume, 0.0f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Speed Woosh Pitch / Sweep", &state.speedWooshPitch, 0.1f, 3.0f, "%.2f");
					modified |= ImGui::SliderFloat("Speed Woosh Resonance", &state.speedWooshResonance, 0.1f, 3.0f, "%.2f");
				}
			}

			if (modified) {
				mgr->SetState(state);
				if (ServiceLocator::Instance().Has<ConfigManager>()) {
					auto cfg = ServiceLocator::Instance().Get<ConfigManager>();
					if (cfg) {
						mgr->SaveState(*cfg);
					}
				}
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
