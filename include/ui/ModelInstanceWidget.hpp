#pragma once

#include <cstdint>
#include <vector>

#include "model/ManifoldBranchingTreeAsset.hpp"
#include "model/ManifoldRockAsset.hpp"
#include "model/ModelInstanceManager.hpp"
#include "model/ProceduralPlantGenerator.hpp"
#include "ui/IWidget.hpp"

namespace brassica::ui {

	// Exercise for the GPU-driven model instance table (model/ModelInstanceManager.hpp) and,
	// since Stage 2, ModelRegistry's multi-model support: spawns a herd of a chosen model's
	// instances with desynchronized clips/startTimes, proving the CPU only ever touches a slot
	// on spawn/clip-change, never per-frame, and that different modelIndex values render
	// correctly side by side.
	class ModelInstanceWidget: public IWidget {
	public:
		ModelInstanceWidget();
		~ModelInstanceWidget() override = default;

		void Draw() override;

		[[nodiscard]] std::string_view GetTitle() const override { return "Model Instances"; }

		[[nodiscard]] std::string_view GetCategory() const override { return "Environment"; }

	private:
		struct SpawnedInstance {
			ModelInstanceHandle handle;
			std::uint32_t       modelIndex;
		};

		std::vector<SpawnedInstance> m_spawnedInstances;
		int                          m_selectedModel{0};
		bool                         m_liveWind{false}; // Stage 4: spawn with kModelInstanceLiveCapableBit set

		// Procedural plant generation v2's ad-hoc designer: one single generated-and-placed
		// model, re-generated in place (ModelRegistry::ReplaceAsset) rather than leaking a new
		// registered model every press -- see ReplaceAsset's own comment for the despawn-before-
		// replace ordering this follows.
		enum class DesignerType { Tree, Bush, Flower, Rock };
		DesignerType         m_designerType{DesignerType::Tree};
		bool                 m_designerHasModel{false};
		std::uint32_t        m_designerModelIndex{0};
		ModelInstanceHandle  m_designerInstance{kInvalidModelInstanceHandle};
		// Tree and Bush both go through BuildProceduralBranchingTree (same generator, different
		// defaults -- see ProceduralArchetypeCatalog.cpp's own identical reasoning), so each gets
		// its own tuned param set rather than sharing one.
		TreeGenParams  m_designerTreeParams;
		TreeGenParams  m_designerBushParams;
		PlantGenParams m_designerFlowerParams;
		RockGenParams  m_designerRockParams;

		void DrawDesignerSection();
	};

} // namespace brassica::ui
