#pragma once
#include <cstdint>

namespace brassica {

	// Frames in flight: how many of the engine's per-frame resources (command buffers, UBOs,
	// descriptor sets) are round-robined so the CPU can record frame N+1 while frame N is still
	// executing on the GPU. Previously redeclared independently in Engine.hpp, DeferredPass.hpp,
	// and AtmosphereLUTPass.hpp -- three constants that had to agree by convention, not by
	// construction.
	inline constexpr std::uint32_t FRAME_OVERLAP = 2;

} // namespace brassica
