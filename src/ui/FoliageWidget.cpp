#include "ui/FoliageWidget.hpp"

#include "ConfigManager.hpp"
#include "foliage/IFoliageManager.hpp"
#include "imgui.h"
#include "ServiceLocator.hpp"

namespace brassica::ui {

	FoliageWidget::FoliageWidget() {
		m_visible = false;
	}

	void FoliageWidget::Draw() {
		if (!m_visible)
			return;

		ImGui::SetNextWindowPos(ImVec2(100.0f, 150.0f), ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(400.0f, 500.0f), ImGuiCond_FirstUseEver);

		if (ImGui::Begin("Foliage & Grass Settings", &m_visible, ImGuiWindowFlags_AlwaysAutoResize)) {
			if (!ServiceLocator::Instance().Has<IFoliageManager>()) {
				ImGui::TextColored(ImVec4(1, 0, 0, 1), "IFoliageManager not available");
				ImGui::End();
				return;
			}

			auto mgr = ServiceLocator::Instance().Get<IFoliageManager>();
			bool enabled = mgr->IsEnabled();
			if (ImGui::Checkbox("Enable Foliage & Grass System", &enabled)) {
				mgr->SetEnabled(enabled);
			}

			if (enabled) {
				ImGui::Separator();
				if (ImGui::CollapsingHeader("Global Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
					auto props = mgr->GetGlobalProperties();
					bool modified = false;

					modified |= ImGui::SliderFloat("Density Multiplier", &props.densityMultiplier, 0.0f, 5.0f, "%.2f");
					modified |= ImGui::SliderFloat("Length Multiplier", &props.lengthMultiplier, 0.1f, 5.0f, "%.2f");
					modified |= ImGui::SliderFloat("Width Multiplier", &props.widthMultiplier, 0.1f, 5.0f, "%.2f");
					modified |= ImGui::SliderFloat("Wind Multiplier", &props.windMultiplier, 0.0f, 5.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rigidity Multiplier", &props.rigidityMultiplier, 0.0f, 2.0f, "%.2f");

					if (modified) {
						mgr->SetGlobalProperties(props);
					}
				}

				if (ImGui::CollapsingHeader("Species Distribution", ImGuiTreeNodeFlags_DefaultOpen)) {
					auto props = mgr->GetGlobalProperties();
					bool modified = false;

					modified |= ImGui::SliderFloat("Flower Ratio", &props.flowerRatio, 0.0f, 1.0f, "%.2f");
					modified |= ImGui::SliderFloat("Fern Ratio", &props.fernRatio, 0.0f, 1.0f, "%.2f");
					modified |= ImGui::SliderFloat("Rock Ratio", &props.rockRatio, 0.0f, 1.0f, "%.2f");
					modified |= ImGui::SliderFloat("Seaweed Ratio", &props.seaweedRatio, 0.0f, 1.0f, "%.2f");
					modified |= ImGui::SliderFloat("Bush Ratio", &props.bushRatio, 0.0f, 1.0f, "%.2f");
					modified |= ImGui::SliderFloat("Tree Ratio", &props.treeRatio, 0.0f, 1.0f, "%.2f");

					if (modified) {
						mgr->SetGlobalProperties(props);
					}
				}

				if (ImGui::CollapsingHeader("LOD Grid & Distance Scale", ImGuiTreeNodeFlags_DefaultOpen)) {
					auto props = mgr->GetGlobalProperties();
					bool modified = false;

					// Jitter-cell size for the per-cell stable-hash placement grid -- previously
					// round-tripped through FoliageManager but never reached the shader at all.
					modified |= ImGui::SliderFloat("Base Scale", &props.baseScale, 0.1f, 2.0f, "%.2f");
					modified |= ImGui::SliderFloat("Base Tile Size", &props.baseTileSize, 4.0f, 128.0f, "%.1f m");
					modified |= ImGui::SliderFloat("LOD Base Range", &props.lodBaseRange, 5.0f, 1000.0f, "%.1f m");
					modified |= ImGui::SliderFloat("LOD Scale Factor", &props.lodScaleFactor, 0.5f, 5.0f, "%.2f");

					int maxLODs = static_cast<int>(props.maxLODs);
					if (ImGui::SliderInt("Max LODs", &maxLODs, 1, 16)) {
						props.maxLODs = static_cast<uint32_t>(maxLODs);
						modified = true;
					}

					int tilesPerRow = static_cast<int>(props.tilesPerRow);
					if (ImGui::SliderInt("Tiles Per Row", &tilesPerRow, 4, 64)) {
						props.tilesPerRow = static_cast<uint32_t>(tilesPerRow);
						modified = true;
					}

					bool occlusionCulling = (props.enableOcclusionCulling != 0u);
					if (ImGui::Checkbox("Terrain Occlusion Culling", &occlusionCulling)) {
						props.enableOcclusionCulling = occlusionCulling ? 1u : 0u;
						modified = true;
					}

					if (modified) {
						mgr->SetGlobalProperties(props);
					}
				}

				if (ImGui::CollapsingHeader("Per-Biome Grass Properties")) {
					static int selectedBiome = 0;
					// Indexed by shaders/helpers/whittaker.glsl's WhittakerBiome::biomeIndex (0-9) --
					// these are the real per-location biomes, not 8 anonymous slots.
					ImGui::Combo(
						"Select Biome",
						&selectedBiome,
						"Ice / Snow\0"
						"Tundra\0"
						"Taiga / Boreal Forest\0"
						"Cold Desert / Grassland\0"
						"Woodland / Shrubland\0"
						"Deciduous Forest\0"
						"Temperate Rainforest\0"
						"Subtropical Desert\0"
						"Savanna\0"
						"Tropical Rainforest\0"
					);

					GrassProperties bProps = mgr->GetBiomeProperties(static_cast<uint32_t>(selectedBiome));
					bool bModified = false;

					bool bEnabled = (bProps.enabled != 0u);
					if (ImGui::Checkbox("Biome Enabled", &bEnabled)) {
						bProps.enabled = bEnabled ? 1u : 0u;
						bModified = true;
					}

					bModified |= ImGui::ColorEdit4("Color Top", &bProps.colorTop.x);
					bModified |= ImGui::ColorEdit4("Color Bottom", &bProps.colorBottom.x);
					bModified |= ImGui::SliderFloat("Height", &bProps.height, 0.1f, 5.0f, "%.2f");
					bModified |= ImGui::SliderFloat("Width", &bProps.width, 0.01f, 0.5f, "%.3f");
					bModified |= ImGui::SliderFloat("Density", &bProps.density, 0.0f, 2.0f, "%.2f");
					bModified |= ImGui::SliderFloat("Flower Ratio##Biome", &bProps.flowerRatio, 0.0f, 1.0f, "%.2f");

					if (bModified) {
						mgr->SetBiomeProperties(static_cast<uint32_t>(selectedBiome), bProps);
					}
				}
			}
		}
		ImGui::End();
	}

} // namespace brassica::ui
