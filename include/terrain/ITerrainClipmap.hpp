#pragma once

#include "constants.h"
#include "IManager.hpp"

namespace brassica {

	struct TerrainClipmapState {
		uint32_t numLODs{constants::Class::Terrain::DefaultMaxLODs};
		float    baseTexelSize{constants::Class::Terrain::BaseTexelSize};

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("numLODs", "Number of LODs", &TerrainClipmapState::numLODs, 1u, 16u, UIHint::Slider),
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
