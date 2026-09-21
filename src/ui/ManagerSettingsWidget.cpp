#include "ui/ManagerSettingsWidget.hpp"

#include <vector>

#include "ArgparseManager.hpp"
#include "ConfigManager.hpp"
#include "IManager.hpp"
#include "imgui.h"
#include "lighting/ILightManager.hpp"
#include "ServiceLocator.hpp"
#include "terrain/ITerrainClipmap.hpp"

namespace brassica::ui {

	ManagerSettingsWidget::ManagerSettingsWidget() {
		m_visible = false;
	}

	void ManagerSettingsWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(100.0f, 100.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Manager Settings", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			ConfigManager* cfg = ServiceLocator::Instance().Has<ConfigManager>()
				? ServiceLocator::Instance().Get<ConfigManager>().get()
				: nullptr;

			std::vector<IManager*> managers;

			if (ServiceLocator::Instance().Has<ILightManager>()) {
				managers.push_back(ServiceLocator::Instance().Get<ILightManager>().get());
			}
			if (ServiceLocator::Instance().Has<ITerrainClipmap>()) {
				managers.push_back(ServiceLocator::Instance().Get<ITerrainClipmap>().get());
			}
			if (ServiceLocator::Instance().Has<ArgparseManager>()) {
				managers.push_back(ServiceLocator::Instance().Get<ArgparseManager>().get());
			}

			for (auto* mgr : managers) {
				if (!mgr)
					continue;
				if (ImGui::CollapsingHeader(mgr->GetManagerName().c_str())) {
					if (mgr->DrawUI()) {
						if (cfg) {
							mgr->SaveState(*cfg);
						}
					}
				}
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
