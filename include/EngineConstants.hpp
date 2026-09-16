#pragma once
#include <cstdint>

#include "constants.h"

namespace brassica {

	// Frames in flight: how many of the engine's per-frame resources (command buffers, UBOs,
	// descriptor sets) are round-robined so the CPU can record frame N+1 while frame N is still
	// executing on the GPU.
	inline constexpr std::uint32_t FRAME_OVERLAP = constants::Engine::FrameOverlap;

	// Scale planet radius: 600km = 600,000 units (1/10th scale planet)
	inline constexpr float FAKE_PLANET_RADIUS = constants::Engine::FakePlanetRadius;

} // namespace brassica
