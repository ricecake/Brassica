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
			glm::vec3 rawDir(dir.x, dir.z, -dir.y);
			rawDir /= (std::abs(rawDir.x) + std::abs(rawDir.y) + std::abs(rawDir.z) + 1e-6f);
			glm::vec2 oct = (rawDir.y >= 0.0f) ? glm::vec2(rawDir.x, rawDir.z) : octWrap(glm::vec2(rawDir.x, rawDir.z));
			return oct * 0.5f + 0.5f;
		}

		inline glm::vec3 octahedralUVToDirection(glm::vec2 uv) {
			glm::vec2 oct = uv * 2.0f - 1.0f;
			glm::vec3 rawDir = glm::vec3(oct.x, 1.0f - std::abs(oct.x) - std::abs(oct.y), oct.y);
			if (rawDir.y < 0.0f) {
				glm::vec2 wrapped = octWrap(glm::vec2(rawDir.x, rawDir.z));
				rawDir.x = wrapped.x;
				rawDir.z = wrapped.y;
			}
			glm::vec3 nRaw = glm::normalize(rawDir);
			return glm::vec3(nRaw.x, -nRaw.z, nRaw.y);
		}

		// Maps octahedral UV coordinates that have strayed outside [0, 1]^2 back onto the valid
		// octahedron net. The net's outer edges are fold lines of the octahedron, not a periodic
		// boundary, so a plain modulo wrap (or decoding+re-encoding the out-of-domain UV, which
		// extrapolates octahedralUVToDirection past its valid piecewise-linear domain) produces a
		// position with an uncontrolled lateral offset. Mirroring across the crossed edge is the
		// correct, continuous continuation (verified: octahedralUVToDirection(1, y) ==
		// octahedralUVToDirection(1, 1 - y), etc. for all four edges).
		inline glm::vec2 wrapOctahedralUV(glm::vec2 uv) {
			if (uv.x < 0.0f) {
				uv.x = -uv.x;
				uv.y = 1.0f - uv.y;
			} else if (uv.x > 1.0f) {
				uv.x = 2.0f - uv.x;
				uv.y = 1.0f - uv.y;
			}
			if (uv.y < 0.0f) {
				uv.y = -uv.y;
				uv.x = 1.0f - uv.x;
			} else if (uv.y > 1.0f) {
				uv.y = 2.0f - uv.y;
				uv.x = 1.0f - uv.x;
			}
			return uv;
		}

	} // namespace octahedral

} // namespace brassica
