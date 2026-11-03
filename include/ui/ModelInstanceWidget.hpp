#pragma once

#include <cstdint>
#include <vector>

#include "model/ModelInstanceManager.hpp"
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
	};

} // namespace brassica::ui
