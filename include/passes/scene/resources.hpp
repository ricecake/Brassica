#pragma once

// Scene-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes.

#include <algorithm>
#include <cmath>

#include "vulkan/vulkan.hpp"

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"

namespace brassica {

	inline graph::ResourceDesc HiZTextureDesc(std::uint32_t width, std::uint32_t height, std::uint32_t mips = 0) {
		std::uint32_t calculatedMips = mips;
		if (calculatedMips == 0) {
			const std::uint32_t maxDim = (std::max)(width, height);
			calculatedMips = maxDim > 0
				? static_cast<std::uint32_t>(std::floor(std::log2(static_cast<float>(maxDim)))) + 1
				: 1;
		}
		calculatedMips = (std::min)(calculatedMips, 12u);

		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.mips = calculatedMips,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	// First production conversion to the type-owned-desc model (see graph/ResourceCatalog.hpp):
	// Desc() is the exact auto-compute path HiZTextureDesc(w, h) already took (mips=0), which
	// Spike A's desc census confirmed is arithmetically identical to HiZDownsampleNode's own
	// explicit-mips call -- both reduce to floor(log2(maxDim))+1, clamped to 12. kLifetime =
	// History documents that this type maintains an interframe double-buffered pair (today still
	// via the History<HiZTexture> key wrapper, not yet read from this field anywhere -- see
	// ResourceKey.hpp's Lifetime comment).
	struct HiZTexture {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::History;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return HiZTextureDesc(ctx.width, ctx.height);
		}
	};

	// Real device-negotiated VRS tile size, clamped into the device's reported
	// min/maxFragmentShadingRateAttachmentTexelSize range by Engine::InitVulkan after querying
	// VkPhysicalDeviceFragmentShadingRatePropertiesKHR -- constants::Class::ShadingRate::
	// TileWidth/TileHeight (16x16) is only the compile-time default these start at. Confirmed via
	// real-device validation that logging a mismatch and then using the wrong size anyway still
	// fails: a GPU reporting an 8x8-only range (min == max == 8, seen on real hardware) rejects a
	// 16x16 attachment outright at vkCmdBeginRendering, so every consumer of the tile size --
	// ShadingRateMap's own Desc(), ShadingRateComputeNode's dispatch grid, and every
	// ReadPrevious<ShadingRateMap>().ShadingRateTexelSize() call -- has to read these, not the
	// bare constants, for VRS to work on a device whose supported size differs from the default.
	inline std::uint32_t s_shadingRateTileWidth = constants::Class::ShadingRate::TileWidth;
	inline std::uint32_t s_shadingRateTileHeight = constants::Class::ShadingRate::TileHeight;

	// One tile (s_shadingRateTileWidth/s_shadingRateTileHeight) per texel, holding a
	// VK_KHR_fragment_shading_rate-encoded rate byte (ShadingRateComputeNode/shading_rate.comp
	// writes it, DynamicRenderingWrapper::Begin -- PhysicalExecutionBackend.hpp -- binds it as an
	// attachment automatically for any Graphics node that does ReadPrevious<ShadingRateMap>()).
	// History lifetime, same ping-ponged double-buffer HiZTexture above already uses: this
	// frame's compute pass reads last frame's map (the dither-refresh schedule's own input) while
	// writing this frame's.
	struct ShadingRateMap {
		static constexpr graph::Lifetime kLifetime = graph::Lifetime::History;

		[[nodiscard]] graph::ResourceDesc Desc(const graph::FrameContext& ctx) const {
			return graph::ShadingRateAttachmentDesc(
				(ctx.width + s_shadingRateTileWidth - 1) / s_shadingRateTileWidth,
				(ctx.height + s_shadingRateTileHeight - 1) / s_shadingRateTileHeight
			);
		}
	};

} // namespace brassica
