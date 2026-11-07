#include "ui/TextureDumpWidget.hpp"

#include "imgui.h"
#include "ServiceLocator.hpp"
#include "TextureDumpService.hpp"

namespace brassica::ui {

	void TextureDumpWidget::Draw() {
		if (!m_visible) {
			return;
		}

		ImGui::SetNextWindowPos(ImVec2(100.0f, 150.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(280.0f, 240.0f), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Texture Dump", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			if (!ServiceLocator::Instance().Has<TextureDumpService>()) {
				ImGui::TextColored(ImVec4(1, 0, 0, 1), "TextureDumpService not available");
				ImGui::End();
				return;
			}

			auto service = ServiceLocator::Instance().Get<TextureDumpService>();
			ImGui::TextUnformatted("Dumps to debug_dumps/ one at a time:");
			ImGui::Separator();

			for (const TextureDumpEntry& entry : service->Entries()) {
				if (ImGui::Button(entry.label.c_str())) {
					bool ok = service->Dump(entry, "debug_dumps");
					m_lastResult = entry.label + (ok ? ": saved" : ": FAILED (see log)");
				}
			}

			if (!m_lastResult.empty()) {
				ImGui::Separator();
				ImGui::TextUnformatted(m_lastResult.c_str());
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
