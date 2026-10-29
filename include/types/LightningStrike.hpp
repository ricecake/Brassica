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
	struct alignas(16) LightningStrikeSlot {
		glm::vec4 originAndSeed{0.0f}; // xyz = cloud-side origin world pos, w = PRNG seed
		glm::vec4 endPos{0.0f};        // xyz = second cloud-side point, w = unused
		float     lifetime{0.0f};
		float     maxLifetime{0.0f};
		float     _pad0{0.0f};
		float     _pad1{0.0f};
	};

	static_assert(sizeof(LightningStrikeSlot) == 48, "LightningStrikeSlot must be 48 bytes (std430)");

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
