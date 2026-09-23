#pragma once

#include "IManager.hpp"

namespace brassica {

	struct TerrainClipmapState {
		uint32_t numLODs{10};
		float    baseTexelSize{0.5f};

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

		virtual bool GetHeightAtWorldPos(float worldX, float worldZ, float& outHeight) const {
			(void)worldX;
			(void)worldZ;
			outHeight = 0.0f;
			return false;
		}

		virtual float SampleHeight(float worldX, float worldZ) const {
			(void)worldX;
			(void)worldZ;
			return 0.0f;
		}

		virtual bool ExportTerrainMapPNG(
			const std::string& filepath,
			glm::vec2          centerWorldPos,
			float              chunkExtent,
			uint32_t           resolution
		) const {
			(void)filepath;
			(void)centerWorldPos;
			(void)chunkExtent;
			(void)resolution;
			return false;
		}
	};

} // namespace brassica
