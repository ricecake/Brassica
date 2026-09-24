#pragma once

#include <algorithm>
#include <cmath>
#include "constants.h"
#include "IManager.hpp"

namespace brassica {

	inline uint32_t CalculateBaseLOD(float height, uint32_t totalLODs, uint32_t activeWindow) {
		constexpr float baseHeightThreshold = 50.0f;
		if (height <= baseHeightThreshold) {
			return 0;
		}
		float factor = height / baseHeightThreshold;
		int calculatedLOD = static_cast<int>(std::floor(std::log2(factor)));
		int maxBaseLOD = static_cast<int>(totalLODs) - static_cast<int>(activeWindow);
		if (maxBaseLOD < 0) maxBaseLOD = 0;
		return static_cast<uint32_t>(std::clamp(calculatedLOD, 0, maxBaseLOD));
	}

	struct TerrainClipmapState {
		uint32_t numLODs{constants::Class::Terrain::DefaultMaxLODs};
		uint32_t windowLODs{constants::Class::Terrain::DefaultWindowLODs};
		float    baseTexelSize{constants::Class::Terrain::BaseTexelSize};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("numLODs", "Number of LODs", &TerrainClipmapState::numLODs, 10u, 23u, UIHint::Slider),
				MakeField("windowLODs", "Window LODs", &TerrainClipmapState::windowLODs, 10u, 12u, UIHint::Slider),
				MakeField(
					"baseTexelSize",
					"Base Texel Size",
					&TerrainClipmapState::baseTexelSize,
					0.01f,
					10.0f,
					UIHint::Slider
				)
			);
		}
	};

	class ITerrainClipmap: public ManagerBase<ITerrainClipmap, TerrainClipmapState> {
	public:
		using State = TerrainClipmapState;

		~ITerrainClipmap() override = default;

		std::string GetManagerName() const override { return "TerrainClipmap"; }

		State GetState() const override { return State{}; }

		void SetState(const State& state) override { (void)state; }

		virtual void Regenerate() = 0;
	};

} // namespace brassica
