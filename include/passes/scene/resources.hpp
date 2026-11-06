#pragma once

// Scene-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes.

#include <algorithm>
#include <cmath>

#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"

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

} // namespace brassica
