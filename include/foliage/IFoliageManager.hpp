#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <glm/glm.hpp>

#include "IManager.hpp"
#include "ManagerReflection.hpp"

namespace brassica {

	// Matches shaders/helpers/whittaker.glsl's WhittakerBiome::biomeIndex range (0-9) exactly --
	// this used to be 8 anonymous slots with no relation to the real per-location biome signal,
	// which only ever comes from the weather texture (TerrainBiomeTexture's own per-location
	// value is hardcoded to 0 in terrain_gen.comp).
	inline constexpr std::uint32_t kFoliageBiomeCount = 10;

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
		float     clumpDensity{1.0f};
		float     bladeDensity{1.0f};
	};

	// Large Foliage Page Table entry in SSBO (std430 layout)
	struct LargeFoliagePageTableEntry {
		std::uint32_t meshletOffset{0};   // Offset into LargeFoliageMeshletBuffer
		std::uint32_t meshletCount{0};    // Number of meshlets for this model variety
		std::uint32_t vertexOffset{0};    // Vertex offset in SSBO
		std::uint32_t primitiveOffset{0}; // Primitive/index offset in SSBO
		glm::vec4     scaleAndRadius{1.0f, 1.0f, 1.0f, 2.0f}; // x,y,z scale / radius
		glm::uvec4    varietyParams{0u, 0u, 0u, 0u}; // x = modelVariety (0 Pine, 1 Palm, 2 Deciduous, 3 Cactus)
	};

	// Meshlet vertex format for Large Foliage stored in SSBO
	struct LargeFoliageVertexGPU {
		glm::vec4 positionAndU{0.0f};                // xyz = local position, w = u
		glm::vec4 normalAndV{0.0f, 1.0f, 0.0f, 0.0f}; // xyz = local normal, w = v
		glm::vec4 color{1.0f};                       // rgba color
		glm::uvec4 materialAndFlags{6u, 0u, 0u, 0u}; // x = material type
	};

	// Large Foliage Meshlet header in SSBO
	struct LargeFoliageMeshletGPU {
		std::uint32_t vertexOffset{0};
		std::uint32_t vertexCount{0};
		std::uint32_t primitiveOffset{0};
		std::uint32_t primitiveCount{0};
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
		float    rockRatio{0.05f};
		float    seaweedRatio{0.05f};
		float    bushRatio{0.10f};
		float    treeRatio{0.10f};
		float    baseTileSize{16.0f};
		uint32_t maxLODs{8};
		uint32_t tilesPerRow{16};
		uint32_t enableOcclusionCulling{1};
		uint32_t enableHiZCulling{1};
		float    clumpDensity{1.0f};
		float    bladeDensity{1.0f};
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
		float    rockRatio{0.05f};
		float    seaweedRatio{0.05f};
		float    bushRatio{0.10f};
		float    treeRatio{0.10f};
		float    baseTileSize{16.0f};
		uint32_t maxLODs{8};
		uint32_t tilesPerRow{16};
		bool     enableOcclusionCulling{true};
		bool     enableHiZCulling{true};
		float    clumpDensity{1.0f};
		float    bladeDensity{1.0f};

		auto GetReflection() {
			return std::make_tuple(
				MakeField("enabled", "Enable Foliage", &FoliageState::enabled),
				MakeField(
					"lengthMultiplier",
					"Length Multiplier",
					&FoliageState::lengthMultiplier,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"widthMultiplier",
					"Width Multiplier",
					&FoliageState::widthMultiplier,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"densityMultiplier",
					"Density Multiplier",
					&FoliageState::densityMultiplier,
					0.0f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"rigidityMultiplier",
					"Rigidity Multiplier",
					&FoliageState::rigidityMultiplier,
					0.0f,
					2.0f,
					UIHint::Slider
				),
				MakeField(
					"windMultiplier",
					"Wind Multiplier",
					&FoliageState::windMultiplier,
					0.0f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"clumpDensity",
					"Clump Density",
					&FoliageState::clumpDensity,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"bladeDensity",
					"Blade Density",
					&FoliageState::bladeDensity,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"lodScaleFactor",
					"LOD Scale Factor",
					&FoliageState::lodScaleFactor,
					0.5f,
					5.0f,
					UIHint::Slider
				),
				MakeField("lodBaseRange", "LOD Base Range", &FoliageState::lodBaseRange, 5.0f, 1000.0f, UIHint::Slider),
				MakeField("baseScale", "Base Scale", &FoliageState::baseScale, 0.1f, 2.0f, UIHint::Slider),
				MakeField("flowerRatio", "Flower Ratio", &FoliageState::flowerRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("fernRatio", "Fern Ratio", &FoliageState::fernRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("rockRatio", "Rock Ratio", &FoliageState::rockRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("seaweedRatio", "Seaweed Ratio", &FoliageState::seaweedRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("bushRatio", "Bush Ratio", &FoliageState::bushRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("treeRatio", "Tree Ratio", &FoliageState::treeRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("baseTileSize", "Base Tile Size", &FoliageState::baseTileSize, 4.0f, 128.0f, UIHint::Slider),
				MakeField("maxLODs", "Max LODs", &FoliageState::maxLODs, 1u, 16u, UIHint::Slider),
				MakeField("tilesPerRow", "Tiles Per Row", &FoliageState::tilesPerRow, 4u, 64u, UIHint::Slider),
				MakeField("enableOcclusionCulling", "Terrain Occlusion Culling", &FoliageState::enableOcclusionCulling),
				MakeField("enableHiZCulling", "HiZ Occlusion Culling", &FoliageState::enableHiZCulling)
			);
		}

		auto GetReflection() const {
			return std::make_tuple(
				MakeField("enabled", "Enable Foliage", &FoliageState::enabled),
				MakeField(
					"lengthMultiplier",
					"Length Multiplier",
					&FoliageState::lengthMultiplier,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"widthMultiplier",
					"Width Multiplier",
					&FoliageState::widthMultiplier,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"densityMultiplier",
					"Density Multiplier",
					&FoliageState::densityMultiplier,
					0.0f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"rigidityMultiplier",
					"Rigidity Multiplier",
					&FoliageState::rigidityMultiplier,
					0.0f,
					2.0f,
					UIHint::Slider
				),
				MakeField(
					"windMultiplier",
					"Wind Multiplier",
					&FoliageState::windMultiplier,
					0.0f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"clumpDensity",
					"Clump Density",
					&FoliageState::clumpDensity,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"bladeDensity",
					"Blade Density",
					&FoliageState::bladeDensity,
					0.1f,
					5.0f,
					UIHint::Slider
				),
				MakeField(
					"lodScaleFactor",
					"LOD Scale Factor",
					&FoliageState::lodScaleFactor,
					0.5f,
					5.0f,
					UIHint::Slider
				),
				MakeField("lodBaseRange", "LOD Base Range", &FoliageState::lodBaseRange, 5.0f, 1000.0f, UIHint::Slider),
				MakeField("baseScale", "Base Scale", &FoliageState::baseScale, 0.1f, 2.0f, UIHint::Slider),
				MakeField("flowerRatio", "Flower Ratio", &FoliageState::flowerRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("fernRatio", "Fern Ratio", &FoliageState::fernRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("rockRatio", "Rock Ratio", &FoliageState::rockRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("seaweedRatio", "Seaweed Ratio", &FoliageState::seaweedRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("bushRatio", "Bush Ratio", &FoliageState::bushRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("treeRatio", "Tree Ratio", &FoliageState::treeRatio, 0.0f, 1.0f, UIHint::Slider),
				MakeField("baseTileSize", "Base Tile Size", &FoliageState::baseTileSize, 4.0f, 128.0f, UIHint::Slider),
				MakeField("maxLODs", "Max LODs", &FoliageState::maxLODs, 1u, 16u, UIHint::Slider),
				MakeField("tilesPerRow", "Tiles Per Row", &FoliageState::tilesPerRow, 4u, 64u, UIHint::Slider),
				MakeField("enableOcclusionCulling", "Terrain Occlusion Culling", &FoliageState::enableOcclusionCulling),
				MakeField("enableHiZCulling", "HiZ Occlusion Culling", &FoliageState::enableHiZCulling)
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
		virtual void                                SetGlobalProperties(const GlobalGrassProperties& props) = 0;
	};

} // namespace brassica
