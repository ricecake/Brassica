#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>

#include "IManager.hpp"
#include "ManagerReflection.hpp"

namespace brassica {

	struct GrassProperties {
		glm::vec4 colorTop{0.3f, 0.8f, 0.2f, 1.0f};
		glm::vec4 colorBottom{0.1f, 0.3f, 0.05f, 1.0f};
		float     height{1.0f};
		float     width{0.1f};
		float     rigidity{0.5f};
		float     heightVariance{0.3f};
		float     widthVariance{0.1f};
		float     density{1.0f};
		float     colorVariability{0.2f};
		float     windInfluence{1.0f};
		uint32_t  enabled{1};
		float     flowerRatio{0.1f};
	};

	struct GlobalGrassProperties {
		float    lengthMultiplier{1.0f};
		float    widthMultiplier{1.0f};
		float    densityMultiplier{1.0f};
		float    rigidityMultiplier{1.0f};
		float    windMultiplier{1.0f};
		uint32_t enabled{1};
		float    lodScaleFactor{2.0f};
		float    lodBaseRange{20.0f};
		float    baseScale{0.5f};
		float    flowerRatio{0.15f};
		float    fernRatio{0.20f};
		float    baseTileSize{16.0f};
	};

	struct FoliageState {
		bool     enabled{true};
		float    lengthMultiplier{1.0f};
		float    widthMultiplier{1.0f};
		float    densityMultiplier{1.0f};
		float    rigidityMultiplier{1.0f};
		float    windMultiplier{1.0f};
		float    lodScaleFactor{2.0f};
		float    lodBaseRange{20.0f};
		float    baseScale{0.5f};
		float    flowerRatio{0.15f};
		float    fernRatio{0.20f};
		float    baseTileSize{16.0f};
		uint32_t maxLODs{8};
		uint32_t tilesPerRow{16};

		auto GetReflection() {
			return std::make_tuple(
				MakeField("enabled", "Enable Foliage", &FoliageState::enabled),
				MakeField("lengthMultiplier", "Length Multiplier", &FoliageState::lengthMultiplier, 0.1f, 5.0f, UIHint::Slider),
				MakeField("widthMultiplier", "Width Multiplier", &FoliageState::widthMultiplier, 0.1f, 5.0f, UIHint::Slider),
				MakeField("densityMultiplier", "Density Multiplier", &FoliageState::densityMultiplier, 0.0f, 5.0f, UIHint::Slider),
				MakeField("rigidityMultiplier", "Rigidity Multiplier", &FoliageState::rigidityMultiplier, 0.0f, 2.0f, UIHint::Slider),
				MakeField("windMultiplier", "Wind Multiplier", &FoliageState::windMultiplier, 0.0f, 5.0f, UIHint::Slider),
				MakeField("lodScaleFactor", "LOD Scale Factor", &FoliageState::lodScaleFactor, 0.5f, 5.0f, UIHint::Slider),
				MakeField("lodBaseRange", "LOD Base Range", &FoliageState::lodBaseRange, 5.0f, 100.0f, UIHint::Slider),
				MakeField("baseScale", "Base Scale", &FoliageState::baseScale, 0.1f, 2.0f, UIHint::Slider),
				MakeField("flowerRatio", "Flower Ratio", &FoliageState::flowerRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("fernRatio", "Fern Ratio", &FoliageState::fernRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("baseTileSize", "Base Tile Size", &FoliageState::baseTileSize, 4.0f, 64.0f, UIHint::Slider),
				MakeField("maxLODs", "Max LODs", &FoliageState::maxLODs, 1u, 12u, UIHint::Slider),
				MakeField("tilesPerRow", "Tiles Per Row", &FoliageState::tilesPerRow, 4u, 32u, UIHint::Slider)
			);
		}

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("enabled", "Enable Foliage", &FoliageState::enabled),
				MakeField("lengthMultiplier", "Length Multiplier", &FoliageState::lengthMultiplier, 0.1f, 5.0f, UIHint::Slider),
				MakeField("widthMultiplier", "Width Multiplier", &FoliageState::widthMultiplier, 0.1f, 5.0f, UIHint::Slider),
				MakeField("densityMultiplier", "Density Multiplier", &FoliageState::densityMultiplier, 0.0f, 5.0f, UIHint::Slider),
				MakeField("rigidityMultiplier", "Rigidity Multiplier", &FoliageState::rigidityMultiplier, 0.0f, 2.0f, UIHint::Slider),
				MakeField("windMultiplier", "Wind Multiplier", &FoliageState::windMultiplier, 0.0f, 5.0f, UIHint::Slider),
				MakeField("lodScaleFactor", "LOD Scale Factor", &FoliageState::lodScaleFactor, 0.5f, 5.0f, UIHint::Slider),
				MakeField("lodBaseRange", "LOD Base Range", &FoliageState::lodBaseRange, 5.0f, 100.0f, UIHint::Slider),
				MakeField("baseScale", "Base Scale", &FoliageState::baseScale, 0.1f, 2.0f, UIHint::Slider),
				MakeField("flowerRatio", "Flower Ratio", &FoliageState::flowerRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("fernRatio", "Fern Ratio", &FoliageState::fernRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("baseTileSize", "Base Tile Size", &FoliageState::baseTileSize, 4.0f, 64.0f, UIHint::Slider),
				MakeField("maxLODs", "Max LODs", &FoliageState::maxLODs, 1u, 12u, UIHint::Slider),
				MakeField("tilesPerRow", "Tiles Per Row", &FoliageState::tilesPerRow, 4u, 32u, UIHint::Slider)
			);
		}
	};

	class IFoliageManager: public ManagerBase<IFoliageManager, FoliageState> {
	public:
		~IFoliageManager() override = default;

		[[nodiscard]] std::string GetManagerName() const override { return "Foliage Manager"; }

		[[nodiscard]] virtual bool IsEnabled() const = 0;
		virtual void               SetEnabled(bool enabled) = 0;

		[[nodiscard]] virtual const GrassProperties& GetBiomeProperties(uint32_t biomeIdx) const = 0;
		virtual void SetBiomeProperties(uint32_t biomeIdx, const GrassProperties& props) = 0;

		[[nodiscard]] virtual GlobalGrassProperties GetGlobalProperties() const = 0;
		virtual void SetGlobalProperties(const GlobalGrassProperties& props) = 0;
	};

} // namespace brassica
