#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct DownsamplePushConstants {
		glm::vec2     srcResolution{0.0f, 0.0f};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t depthIndex{0};
		std::uint32_t outMip0Index{0};
		std::uint32_t outMip1Index{0};
		std::uint32_t outMip2Index{0};
		std::uint32_t outMip3Index{0};
		std::uint32_t outMip4Index{0};
		std::uint32_t outExpMip0Index{0};
		std::uint32_t outExpMip1Index{0};
		std::uint32_t outExpMip2Index{0};
		std::uint32_t outExpMip3Index{0};
		std::uint32_t outExpMip4Index{0};
		std::uint32_t outWgtMip0Index{0};
		std::uint32_t outWgtMip1Index{0};
		std::uint32_t outWgtMip2Index{0};
		std::uint32_t outWgtMip3Index{0};
		std::uint32_t outWgtMip4Index{0};
		std::int32_t  numMips{5};
		float         threshold{1.0f};
		float         deltaTime{0.016f};
		float         softness{0.5f};
		float         karisDampening{1.0f};
		float         maxPoolingFactor{0.0f};
		// Appended last, deliberately: every scalar field in this struct, same reasoning as
		// TonemapPushConstants.hpp's bloomEnabled comment -- inserting a new field anywhere earlier
		// shifts every C++ offset after it with no compensating padding. Bindless index into
		// CloudAccumulatedTransmittance -- sampled directly (a plain texture, no stencil involved)
		// so the Cloud exposure layer's weight can be the real local cloud opacity instead of a
		// hard binary classification. Replaces this field's old stencilIndex purpose: the binary
		// stencil signal turned out to produce a visible seam at translucent cloud edges, since it
		// snapped the whole pixel to 100% Cloud-layer treatment the instant any haze was detected.
		std::uint32_t cloudTransmittanceIndex{0};
	};

	static_assert(
		sizeof(DownsamplePushConstants) == 104,
		"DownsamplePushConstants size must be 104 bytes -- must match bloom_downsample.comp's push_constant block "
		"exactly"
	);

	// Bloom-wide tunables that aren't per-frame derived (unlike the resource indices/resolution
	// above, which TonemapComputeNode::Execute overwrites every frame) -- exposed as a global so a
	// settings UI can bind directly to it, the same way TonemapPushConstants.hpp's s_tonemapPush
	// is bound.
	inline DownsamplePushConstants s_bloomDownsamplePush{};

	struct BloomUpsamplePushConstants {
		std::uint32_t srcIndex{0};      // coarser mip, sampled bilinearly
		std::uint32_t dstReadIndex{0};  // finer mip, sampled to read its existing (already-downsampled) content
		std::uint32_t dstWriteIndex{0}; // finer mip, storage-image target for the accumulated result
		std::uint32_t _pad0{0};
		glm::vec2     srcResolution{0.0f, 0.0f};
		glm::vec2     dstResolution{0.0f, 0.0f};
		float         filterRadius{1.0f};
	};

	static_assert(
		sizeof(BloomUpsamplePushConstants) == 36,
		"BloomUpsamplePushConstants size must be 36 bytes -- must match bloom_upsample.comp's push_constant block "
		"exactly"
	);

} // namespace brassica
