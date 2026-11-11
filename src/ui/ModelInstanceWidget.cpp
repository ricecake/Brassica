#include "ui/ModelInstanceWidget.hpp"

#include <random>

#include "imgui.h"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/ProceduralArchetypeCatalog.hpp"
#include "ServiceLocator.hpp"
#include "passes/terrain/TerrainManager.hpp"
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

		// Bush starts from different defaults than Tree -- shorter segments, more branches,
		// wider pitch, lower rigidity -- matching ProceduralArchetypeCatalog.cpp's own
		// BushVariantA(), even though both go through the exact same BuildProceduralPlant
		// generator.
		m_designerBushParams.maxSegments = 20;
		m_designerBushParams.segmentLength = 0.14f;
		m_designerBushParams.directionJitterDegrees = 16.0f;
		m_designerBushParams.branchChance = 0.5f;
		m_designerBushParams.minBranchesPerNode = 3;
		m_designerBushParams.maxBranchesPerNode = 4;
		m_designerBushParams.trunkRadiusLow = 0.08f;
		m_designerBushParams.trunkRadiusHigh = 0.06f;
		m_designerBushParams.radiusFalloffPerSegment = 0.85f;
		m_designerBushParams.branchPitchMinDegrees = 45.0f;
		m_designerBushParams.branchPitchMaxDegrees = 85.0f;
		m_designerBushParams.baseRigidity = 0.6f;
		m_designerBushParams.rigidityFalloffPerSegment = 0.10f;
		m_designerBushParams.circularSegments = 8;
		m_designerBushParams.maxJoints = 18;

		m_designerFlowerParams.maxSegments = 5;
		m_designerFlowerParams.segmentLength = 0.14f;
		m_designerFlowerParams.branchChance = 0.0f;
		m_designerFlowerParams.baseRigidity = 0.3f;
		m_designerFlowerParams.enableFlowers = true;
		m_designerFlowerParams.flowerChance = 1.0f;
		m_designerFlowerParams.flowerSize = 0.15f;
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
			std::string currentLabel = registry->GetName(static_cast<std::uint32_t>(m_selectedModel));
			if (currentLabel.empty()) {
				currentLabel = "Model " + std::to_string(m_selectedModel);
			}
			if (ImGui::BeginCombo("Model", currentLabel.c_str())) {
				for (std::uint32_t m = 0; m < modelCount; ++m) {
					std::string label = registry->GetName(m);
					if (label.empty()) {
						label = "Model " + std::to_string(m);
					}
					label += " (" + std::to_string(registry->GetClipCount(m)) + " clips)";
					bool isSelected = (m_selectedModel == static_cast<int>(m));
					if (ImGui::Selectable(label.c_str(), isSelected)) {
						m_selectedModel = static_cast<int>(m);
					}
					if (isSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
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

			ImGui::Separator();
			DrawDesignerSection();
		}
		ImGui::End();
	}

	// Procedural plant generation v2's ad-hoc designer: pick a type, tune a few real parameters,
	// Generate places one instance at the camera's position at ground height (same pattern
	// "Spawn herd" above already uses); pressing Generate again regenerates that same model in
	// place (ModelRegistry::ReplaceAsset) instead of leaking a new registered model every press.
	void ModelInstanceWidget::DrawDesignerSection() {
		ImGui::Text("Ad-Hoc Procedural Designer");

		if (!ServiceLocator::Instance().Has<ModelInstanceManager>() || !ServiceLocator::Instance().Has<ModelRegistry>()) {
			ImGui::TextColored(ImVec4(1, 0, 0, 1), "ModelInstanceManager/ModelRegistry not available");
			return;
		}
		auto instanceMgr = ServiceLocator::Instance().Get<ModelInstanceManager>();
		auto registry = ServiceLocator::Instance().Get<ModelRegistry>();

		const char* typeNames[] = {"Tree", "Bush", "Flower", "Rock"};
		int         typeIdx = static_cast<int>(m_designerType);
		if (ImGui::Combo("Type", &typeIdx, typeNames, IM_ARRAYSIZE(typeNames))) {
			m_designerType = static_cast<DesignerType>(typeIdx);
		}

		switch (m_designerType) {
		case DesignerType::Tree:
			ImGui::InputScalar("Seed##tree", ImGuiDataType_U32, &m_designerTreeParams.seed);
			ImGui::SliderInt("Max Segments", &m_designerTreeParams.maxSegments, 4, 40);
			ImGui::SliderFloat("Branch Chance", &m_designerTreeParams.branchChance, 0.0f, 1.0f);
			ImGui::SliderInt("Min Branches", &m_designerTreeParams.minBranchesPerNode, 1, 5);
			ImGui::SliderInt("Max Branches", &m_designerTreeParams.maxBranchesPerNode, 1, 6);
			ImGui::SliderFloat("Segment Length", &m_designerTreeParams.segmentLength, 0.1f, 0.8f);
			ImGui::SliderFloat("Trunk Radius", &m_designerTreeParams.trunkRadiusLow, 0.02f, 0.6f);
			ImGui::SliderFloat("Base Rigidity", &m_designerTreeParams.baseRigidity, 0.0f, 1.0f);
			break;
		case DesignerType::Bush:
			ImGui::InputScalar("Seed##bush", ImGuiDataType_U32, &m_designerBushParams.seed);
			ImGui::SliderInt("Max Segments##bush", &m_designerBushParams.maxSegments, 4, 40);
			ImGui::SliderFloat("Branch Chance##bush", &m_designerBushParams.branchChance, 0.0f, 1.0f);
			ImGui::SliderInt("Min Branches##bush", &m_designerBushParams.minBranchesPerNode, 1, 6);
			ImGui::SliderInt("Max Branches##bush", &m_designerBushParams.maxBranchesPerNode, 1, 8);
			ImGui::SliderFloat("Segment Length##bush", &m_designerBushParams.segmentLength, 0.05f, 0.4f);
			ImGui::SliderFloat("Trunk Radius##bush", &m_designerBushParams.trunkRadiusLow, 0.02f, 0.3f);
			ImGui::SliderFloat("Base Rigidity##bush", &m_designerBushParams.baseRigidity, 0.0f, 1.0f);
			break;
		case DesignerType::Flower:
			ImGui::InputScalar("Seed##flower", ImGuiDataType_U32, &m_designerFlowerParams.seed);
			ImGui::SliderInt("Stem Segments", &m_designerFlowerParams.maxSegments, 2, 10);
			ImGui::SliderFloat("Segment Length", &m_designerFlowerParams.segmentLength, 0.05f, 0.4f);
			ImGui::SliderFloat("Flower Size", &m_designerFlowerParams.flowerSize, 0.05f, 0.5f);
			ImGui::SliderFloat("Stem Rigidity", &m_designerFlowerParams.baseRigidity, 0.0f, 1.0f);
			break;
		case DesignerType::Rock:
			ImGui::InputScalar("Seed##rock", ImGuiDataType_U32, &m_designerRockParams.seed);
			ImGui::SliderFloat("Base Radius", &m_designerRockParams.baseRadius, 0.2f, 3.0f);
			ImGui::SliderInt("Cluster Count", &m_designerRockParams.clusterCount, 1, 4);
			ImGui::SliderFloat("Noise Amplitude", &m_designerRockParams.noiseAmplitude, 0.0f, 0.6f);
			break;
		}

		if (ImGui::Button("Generate")) {
			// Same real bark/leaf/rock/flower materials the real catalog presets use
			// (ProceduralArchetypeCatalog.cpp), instead of the designer's own params silently
			// defaulting to materialIndex 0 (flat white) and enableLeaves=false -- this is the
			// one place those catalog-registered indices get reused outside the catalog itself.
			if (ServiceLocator::Instance().Has<ProceduralArchetypeIndices>()) {
				auto indices = ServiceLocator::Instance().Get<ProceduralArchetypeIndices>();
				m_designerTreeParams.materialIndex = indices->barkMaterialIndex;
				m_designerTreeParams.enableLeaves = true;
				m_designerTreeParams.leafMaterialIndex = indices->leafMaterialIndex;
				m_designerBushParams.materialIndex = indices->bushMaterialIndex;
				m_designerBushParams.enableLeaves = true;
				m_designerBushParams.leafMaterialIndex = indices->leafMaterialIndex;
				m_designerFlowerParams.materialIndex = indices->flowerMaterialIndex;
				m_designerRockParams.materialIndex = indices->rockMaterialIndex;
			}

			ModelAsset asset;
			switch (m_designerType) {
			case DesignerType::Tree:
				asset = BuildProceduralPlant(m_designerTreeParams);
				++m_designerTreeParams.seed;
				break;
			case DesignerType::Bush:
				asset = BuildProceduralPlant(m_designerBushParams);
				++m_designerBushParams.seed;
				break;
			case DesignerType::Flower:
				asset = BuildProceduralPlant(m_designerFlowerParams);
				++m_designerFlowerParams.seed;
				break;
			case DesignerType::Rock:
				asset = BuildProceduralRock(m_designerRockParams);
				++m_designerRockParams.seed;
				break;
			}

			// Despawn before Replace, per ReplaceAsset's own ordering contract -- a live instance
			// must never point at a model index mid-swap.
			if (m_designerInstance != kInvalidModelInstanceHandle) {
				instanceMgr->Despawn(m_designerInstance);
				m_designerInstance = kInvalidModelInstanceHandle;
			}
			if (!m_designerHasModel) {
				m_designerModelIndex = registry->RegisterAsset(asset, "Designer");
				m_designerHasModel = true;
			} else {
				registry->ReplaceAsset(m_designerModelIndex, asset);
			}

			glm::vec3 position(0.0f);
			if (ServiceLocator::Instance().Has<CameraData>()) {
				position = ServiceLocator::Instance().Get<CameraData>()->position;
			}
			float groundY = position.y;
			if (ServiceLocator::Instance().Has<TerrainManager>()) {
				groundY = ServiceLocator::Instance().Get<TerrainManager>()->GetCachedGroundHeight(0.0f) + 1.0f;
			}
			position.y = groundY;
			position.z -= 5.0f; // a few units ahead of the camera, out of the spawn-herd's own area

			m_designerInstance = instanceMgr->Spawn(m_designerModelIndex, position, 1.0f, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), true);
		}

		ImGui::SameLine();
		ImGui::BeginDisabled(m_designerInstance == kInvalidModelInstanceHandle);
		if (ImGui::Button("Remove")) {
			instanceMgr->Despawn(m_designerInstance);
			m_designerInstance = kInvalidModelInstanceHandle;
		}
		ImGui::EndDisabled();
	}

} // namespace brassica::ui
