#pragma once

#include <cstdint>

namespace brassica {

	struct alignas(16) FrameUBO {
		float    time{0.0f};
		uint32_t frameIndex{0};
		uint32_t globalSeed{0};
		uint32_t frameRandom{0};
	};

	static_assert(sizeof(FrameUBO) == 16, "FrameUBO struct size must be 16 bytes");

} // namespace brassica
