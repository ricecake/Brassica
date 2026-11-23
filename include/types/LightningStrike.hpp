#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	// GPU-only strike-pool slot: climate_ca.comp claims and writes a fresh slot when weather
	// conditions trigger a strike; shaders/lightning_age.comp ages it and writes its reserved
	// light-slot entry; shaders/lightning.task plots its polyline and culls dead slots. The CPU
	// never reads or writes individual fields -- this mirrors Particle.hpp/Light.hpp's own
	// convention of a C++ struct that exists only so sizeof() can size the buffer allocation
	// correctly, not because the CPU touches the data.
	// flags bit 0 (LIGHTNING_FLAG_GROUND, see shaders/lighting.glsl): this is a cloud-to-ground
	// strike, not cloud-to-cloud -- endPos is the ground point instead of a second cloud-side
	// point. Bit 1 (LIGHTNING_FLAG_PENDING_GROUND): climate_ca.comp just claimed this as a
	// ground strike but hasn't resolved a real ground height yet (it has no terrain-height
	// source of its own, see its trigger comment) -- lightning_age.comp resolves endPos.y via
	// terrain/generation.glsl's evaluate_terrain_analytical() the next time it ages this slot,
	// then clears the bit. Until cleared, endPos.y is a placeholder, not a real height.
	struct alignas(16) LightningStrikeSlot {
		glm::vec4     originAndSeed{0.0f}; // xyz = flat-world origin, w = PRNG seed
		glm::vec4     endPos{0.0f};        // xyz = flat-world end point, w = unused
		glm::vec4     colorAndPeak{0.0f};  // rgb = base color, a = per-strike peak brightness
		float         lifetime{0.0f};
		float         maxLifetime{0.0f};
		std::uint32_t flags{0};
		std::uint32_t _pad{0};
	};

	static_assert(sizeof(LightningStrikeSlot) == 64, "LightningStrikeSlot must be 64 bytes (std430)");

	inline constexpr std::uint32_t kMaxLightningStrikes = 32;

	// Header + flat slot array, matching the GLSL LightningStrikePool buffer declaration
	// (shaders/lighting.glsl) field-for-field. ringCounter is the lock-free slot-claim counter
	// climate_ca.comp increments via atomicAdd; the CPU only needs this struct's total size.
	struct LightningStrikePoolLayout {
		std::uint32_t      ringCounter{0};
		std::uint32_t      _pad[3]{0, 0, 0};
		LightningStrikeSlot slots[kMaxLightningStrikes];
	};

} // namespace brassica
