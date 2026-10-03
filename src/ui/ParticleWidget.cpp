#include "ui/ParticleWidget.hpp"

#include "ConfigManager.hpp"
#include "imgui.h"
#include "particle/IParticleManager.hpp"
#include "ServiceLocator.hpp"

namespace brassica::ui {

	ParticleWidget::ParticleWidget() {
		m_visible = false;
	}

	void ParticleWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(100.0f, 200.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(350, 300), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Particle System", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			ConfigManager* cfg = ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ConfigManager>()
				? ServiceLocator::Instance().Get<ConfigManager>().get()
				: nullptr;

			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr) {
					bool enabled = mgr->IsEnabled();
					if (ImGui::Checkbox("Enable Particles##Widget", &enabled)) {
						mgr->SetEnabled(enabled);
						if (cfg) {
							mgr->SaveState(*cfg);
						}
					}

					if (enabled) {
						ImGui::Separator();
						ImGui::TextColored(ImVec4(0, 1, 1, 1), "Particle Allotment:");

						int activeCount = static_cast<int>(mgr->GetActiveParticles());
						int maxCount = static_cast<int>(mgr->GetMaxParticles());
						if (ImGui::SliderInt("Particle Count##Widget", &activeCount, 0, maxCount)) {
							mgr->SetActiveParticles(static_cast<uint32_t>(activeCount));
							if (cfg) {
								mgr->SaveState(*cfg);
							}
						}

						float birdProp = mgr->GetBirdProportion();
						float fishProp = mgr->GetFishProportion();
						float fireflyProp = mgr->GetFireflyProportion();

						bool propsChanged = false;
						propsChanged |= ImGui::SliderFloat("Birds Proportion##Widget", &birdProp, 0.0f, 1.0f, "%.2f");
						propsChanged |= ImGui::SliderFloat("Fish Proportion##Widget", &fishProp, 0.0f, 1.0f, "%.2f");
						propsChanged |= ImGui::SliderFloat("Fireflies Proportion##Widget", &fireflyProp, 0.0f, 1.0f, "%.2f");

						if (propsChanged) {
							mgr->SetBirdProportion(birdProp);
							mgr->SetFishProportion(fishProp);
							mgr->SetFireflyProportion(fireflyProp);
							if (cfg) {
								mgr->SaveState(*cfg);
							}
						}

						ImGui::Separator();
						ImGui::TextColored(ImVec4(0, 1, 1, 1), "Lighting:");

						bool enableLights = mgr->GetEnableLights();
						if (ImGui::Checkbox("Enable Particle Lights##Widget", &enableLights)) {
							mgr->SetEnableLights(enableLights);
							if (cfg) {
								mgr->SaveState(*cfg);
							}
						}
					}
				}
			} else {
				ImGui::TextDisabled("ParticleManager is not available.");
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
