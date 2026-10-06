#pragma once
#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "constants.h"

namespace brassica {

	// Frames in flight: how many of the engine's per-frame resources (command buffers, UBOs,
	// descriptor sets) are round-robined so the CPU can record frame N+1 while frame N is still
	// executing on the GPU.
	inline constexpr std::uint32_t FRAME_OVERLAP = constants::Engine::FrameOverlap;

	// Scale planet radius: 600km = 600,000 units (1/10th scale planet)
	inline constexpr float FAKE_PLANET_RADIUS = constants::Engine::FakePlanetRadius;

	namespace octahedral {

		inline glm::vec2 octWrap(glm::vec2 v) {
			return (1.0f - glm::abs(glm::vec2(v.y, v.x))) *
				glm::vec2(v.x >= 0.0f ? 1.0f : -1.0f, v.y >= 0.0f ? 1.0f : -1.0f);
		}

		inline glm::vec2 directionToOctahedralUV(glm::vec3 dir) {
			dir /= (std::abs(dir.x) + std::abs(dir.y) + std::abs(dir.z) + 1e-6f);
			glm::vec2 oct = (dir.y >= 0.0f) ? glm::vec2(dir.x, dir.z) : octWrap(glm::vec2(dir.x, dir.z));
			return oct * 0.5f + 0.5f;
		}

		inline glm::vec3 octahedralUVToDirection(glm::vec2 uv) {
			glm::vec2 oct = uv * 2.0f - 1.0f;
			glm::vec3 dir = glm::vec3(oct.x, 1.0f - std::abs(oct.x) - std::abs(oct.y), oct.y);
			if (dir.y < 0.0f) {
				glm::vec2 wrapped = octWrap(glm::vec2(dir.x, dir.z));
				dir.x = wrapped.x;
				dir.z = wrapped.y;
			}
			return glm::normalize(dir);
		}

	} // namespace octahedral

} // namespace brassica
