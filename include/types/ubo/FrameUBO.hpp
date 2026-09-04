#pragma once

#include <cstdint>

namespace brassica {

	struct alignas(16) FrameUBO {
		float    time{0.0f};
		float    fov{1.2f};
		float    aspectRatio{16.0f / 9.0f};
		uint32_t frameIndex{0};
		uint32_t globalSeed{0};
		uint32_t frameRandom{0};
		uint32_t padding[2]{0, 0};
	};

	static_assert(sizeof(FrameUBO) == 32, "FrameUBO struct size must be 32 bytes");

} // namespace brassica
