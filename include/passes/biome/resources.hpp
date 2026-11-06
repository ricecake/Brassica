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

	struct TerrainWeatherPingPongTexture: WeatherBiomeImage2D<TerrainWeatherPingPongTexture> {
		static constexpr std::uint32_t kDim = constants::Weather::kSimTextureDim;
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

} // namespace brassica
