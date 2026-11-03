#include "ui/ModelInstanceWidget.hpp"

#include <random>

#include "imgui.h"
#include "model/ModelRegistry.hpp"
#include "ServiceLocator.hpp"
#include "terrain/TerrainManager.hpp"
#include "types/CameraData.hpp"

namespace brassica::ui {

	namespace {
		std::mt19937& Rng() {
			static std::mt19937 rng{std::random_device{}()};
			return rng;
		}
	} // namespace

	ModelInstanceWidget::ModelInstanceWidget() {
		m_visible = false;
	}

	void ModelInstanceWidget::Draw() {
		if (!m_visible) {
			return;
		}

		ImGui::SetNextWindowPos(ImVec2(100.0f, 150.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(360.0f, 220.0f), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Model Instances", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			if (!ServiceLocator::Instance().Has<ModelInstanceManager>() || !ServiceLocator::Instance().Has<ModelRegistry>()) {
				ImGui::TextColored(ImVec4(1, 0, 0, 1), "ModelInstanceManager/ModelRegistry not available");
				ImGui::End();
				return;
			}

			auto instanceMgr = ServiceLocator::Instance().Get<ModelInstanceManager>();
			auto registry = ServiceLocator::Instance().Get<ModelRegistry>();

			std::uint32_t modelCount = registry->ModelCount();
			ImGui::Text("Spawned: %zu instances (capacity %u)", m_spawnedInstances.size(), instanceMgr->Capacity());
			ImGui::Text("Models registered: %u", modelCount);

			if (modelCount == 0) {
				ImGui::End();
				return;
			}
			if (m_selectedModel >= static_cast<int>(modelCount)) {
				m_selectedModel = 0;
			}
			for (std::uint32_t m = 0; m < modelCount; ++m) {
				if (m > 0) {
					ImGui::SameLine();
				}
				ImGui::RadioButton(
					("Model " + std::to_string(m) + " (" + std::to_string(registry->GetClipCount(m)) + " clips)").c_str(),
					&m_selectedModel,
					static_cast<int>(m)
				);
			}

			auto modelIndex = static_cast<std::uint32_t>(m_selectedModel);
			std::uint32_t clipCount = registry->GetClipCount(modelIndex);

			ImGui::Checkbox("Live wind (Stage 4 -- no effect on non-skeletal models)", &m_liveWind);

			if (ImGui::Button("Spawn herd (4x4)")) {
				glm::vec3 basePosition(0.0f);
				if (ServiceLocator::Instance().Has<CameraData>()) {
					basePosition = ServiceLocator::Instance().Get<CameraData>()->position;
				}
				float groundY = basePosition.y;
				if (ServiceLocator::Instance().Has<TerrainManager>()) {
					groundY = ServiceLocator::Instance().Get<TerrainManager>()->GetCachedGroundHeight(0.0f) + 1.0f;
				}

				std::uniform_int_distribution<std::uint32_t> clipDist(0, clipCount > 0 ? clipCount - 1 : 0);
				for (int row = 0; row < 4; ++row) {
					for (int col = 0; col < 4; ++col) {
						glm::vec3 position = basePosition;
						position.x += (static_cast<float>(col) - 1.5f) * 4.0f;
						position.z += (static_cast<float>(row) - 1.5f) * 4.0f - 15.0f; // ahead of spawn
						position.y = groundY;

						ModelInstanceHandle handle = instanceMgr->Spawn(modelIndex, position, 1.0f, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), m_liveWind);
						if (handle == kInvalidModelInstanceHandle) {
							continue;
						}
						if (clipCount > 0) {
							instanceMgr->SetClip(handle, clipDist(Rng()), instanceMgr->GetCurrentTimeMs(), true);
						}
						m_spawnedInstances.push_back({handle, modelIndex});
					}
				}
			}

			ImGui::SameLine();

			if (ImGui::Button("Randomize clips/startTimes")) {
				std::uniform_int_distribution<std::uint32_t> offsetDist(0, 10000);
				for (const SpawnedInstance& inst : m_spawnedInstances) {
					std::uint32_t instClipCount = registry->GetClipCount(inst.modelIndex);
					if (instClipCount == 0) {
						continue;
					}
					std::uniform_int_distribution<std::uint32_t> instClipDist(0, instClipCount - 1);
					std::uint32_t                                startTimeMs = instanceMgr->GetCurrentTimeMs() - offsetDist(Rng());
					instanceMgr->SetClip(inst.handle, instClipDist(Rng()), startTimeMs, true);
				}
			}

			ImGui::SameLine();

			if (ImGui::Button("Despawn all")) {
				for (const SpawnedInstance& inst : m_spawnedInstances) {
					instanceMgr->Despawn(inst.handle);
				}
				m_spawnedInstances.clear();
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
