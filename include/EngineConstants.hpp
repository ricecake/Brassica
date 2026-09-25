#pragma once
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "constants.h"

namespace brassica {

	// Frames in flight: how many of the engine's per-frame resources (command buffers, UBOs,
	// descriptor sets) are round-robined so the CPU can record frame N+1 while frame N is still
	// executing on the GPU.
	inline constexpr std::uint32_t FRAME_OVERLAP = constants::Engine::FrameOverlap;

	// Scale planet radius: 600km = 600,000 units (1/10th scale planet)
	inline constexpr float FAKE_PLANET_RADIUS = constants::Engine::FakePlanetRadius;
	inline constexpr float FAKE_PLANET_PERIMETER = 2.0f * constants::General::Math::Pi * FAKE_PLANET_RADIUS;
	inline constexpr float FAKE_PLANET_HALF_PERIMETER = constants::General::Math::Pi * FAKE_PLANET_RADIUS;

	inline glm::vec2 WrapShortestDistance(const glm::vec2& delta) {
		constexpr float L = FAKE_PLANET_PERIMETER;
		return delta - glm::round(delta / L) * L;
	}

	inline glm::vec3 WrapShortestDistance(const glm::vec3& delta) {
		constexpr float L = FAKE_PLANET_PERIMETER;
		glm::vec2 relXZ = glm::vec2(delta.x, delta.z) - glm::round(glm::vec2(delta.x, delta.z) / L) * L;
		return glm::vec3(relXZ.x, delta.y, relXZ.y);
	}

} // namespace brassica
