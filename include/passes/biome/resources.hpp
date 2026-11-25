#pragma once

// Biome-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why. All of these are owned/written
// by TerrainBiomeNode (passes/biome/nodes/TerrainBiomeNode.hpp).

#include "resources/ResourceTypes.hpp"

namespace brassica {

	// TerrainBiomeNode.hpp still uses kSimTextureDim/kCoverageTextureDim/kBiomeTextureDim/
	// kBlendDitherDivisor/kUpdateInterval directly in its push constants and dispatch math --
	// nothing there changes, it just reaches them through this header's own existing include.
	namespace constants::Weather {
		constexpr std::uint32_t kSimTextureDim = 2048;
		constexpr std::uint32_t kCoverageTextureDim = 2048;
		constexpr std::uint32_t kBiomeTextureDim = 2048;
		constexpr std::uint32_t kHydroTextureDim = 2048;
		constexpr std::uint32_t kBlendDitherDivisor = 8;
		constexpr std::uint32_t kUpdateInterval = 300;
		constexpr std::uint32_t kCellsPerUpdate = 512;

		// World-space period (metres) of CloudDetailVolume's periodic tiling, in all 3 axes.
		// Matches the former near cascade's footprint so existing noise-frequency tuning still
		// reads at roughly the same visual scale.
		constexpr float kCloudVolumePeriod = 20000.0f;
	} // namespace constants::Weather

	// TerrainBiomeNode's blended climate/biome-classification output (TerrainBiomeMapA/B's
	// ping-pong target) -- deliberately a distinct key from TerrainBiomeTexture
	// (passes/terrain/resources.hpp), which TerrainGenNode owns and writes per-texel into the
	// terrain clipmap's own layer/mip shape. The two used to collide on one key (both Create<>'d
	// it with different, incompatible descs -- the numLODs-layer shape vs this single-layer
	// kBiomeTextureDim shape -- which silently reprovisioned/retired the resource every frame
	// and left its bindless descriptor racing between the two shapes, confirmed via
	// GPU-assisted validation as "descriptor index never updated").
	struct TerrainBiomeMainTexture: WeatherBiomeImage2D<TerrainBiomeMainTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kBiomeTextureDim;
	};

	struct TerrainWeatherBiomeTexture: WeatherBiomeImage2D<TerrainWeatherBiomeTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kCoverageTextureDim;
	};

	struct TerrainWeatherMapATexture: WeatherBiomeImage2D<TerrainWeatherMapATexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kCoverageTextureDim;
	};

	struct TerrainWeatherMapBTexture: WeatherBiomeImage2D<TerrainWeatherMapBTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kCoverageTextureDim;
	};

	struct TerrainBiomeMapATexture: WeatherBiomeImage2D<TerrainBiomeMapATexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kBiomeTextureDim;
	};

	struct TerrainBiomeMapBTexture: WeatherBiomeImage2D<TerrainBiomeMapBTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kBiomeTextureDim;
	};

	struct TerrainHydroMapATexture: WeatherBiomeImage2D<TerrainHydroMapATexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kHydroTextureDim;
	};

	struct TerrainHydroMapBTexture: WeatherBiomeImage2D<TerrainHydroMapBTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kHydroTextureDim;
	};

	struct TerrainHydroMapTexture: WeatherBiomeImage2D<TerrainHydroMapTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kHydroTextureDim;
	};

	// struct BiomeCaCell {
	// 	float elevation;    // Static height (0.0 to 1.0)
	// 	float airMoisture;  // Moisture carried in the air
	// 	float groundwater;  // Pooled water in soil
	// 	float vegetation;   // Plant coverage (>0 alive, 0 absent, -1 dead, <-1 active dieoff/fire)
	// 	float waterLevel;   // Geology groundwater table elevation
	// 	float slope;        // Directionless steepness [0, 1]
	// 	float substrate;    // Sediment / nutrient capacity [0, 1]
	// 	float aerosol;      // Non-water air particles (CCN / smoke)
	// };
	// static_assert(sizeof(BiomeCaCell) == 32);

	// struct BiomeCaCellBufferA: FixedBuffer<BiomeCaCellBufferA> {
	// 	static constexpr std::uint64_t kByteSize = static_cast<std::uint64_t>(constants::Weather::kSimTextureDim * constants::Weather::kSimTextureDim) * sizeof(BiomeCaCell);
	// 	static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer;
	// };
	// struct BiomeCaCellBufferB: FixedBuffer<BiomeCaCellBufferB> {
	// 	static constexpr std::uint64_t kByteSize = static_cast<std::uint64_t>(constants::Weather::kSimTextureDim * constants::Weather::kSimTextureDim) * sizeof(BiomeCaCell);
	// 	static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer;
	// };
} // namespace brassica
