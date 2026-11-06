#pragma once

// Atmosphere-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	// SkyBackgroundNode's own backdrop (DeferredNode's compositing fallback wherever no opaque
	// surface was rendered). Same screen-relative shape as HdrColor (resources/core.hpp).
	struct AtmosphereRadiance: ScreenImage2D<AtmosphereRadiance> {
		static constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eColorAttachment |
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
	};
	// Lifetime::Persistent -- TransmittanceLUTNode/MultiScatteringLUTNode (atmosphere/nodes/
	// AtmosphereLUTNode.hpp) only regenerate these when AtmospherePushConstants actually changes
	// (AtmosphereRegenerationState's throttle), so "this frame's producer is inactive" is the
	// *normal*, expected case once the atmosphere settles, not "there's no valid data" -- unlike
	// a genuine feature toggle (GTAO, Foliage, bloom), where inactive really does mean "no
	// contribution wanted." Without this, §E's zero-init list (RelationalCompiler.hpp) can't
	// tell the two apart and wrongly clears a perfectly good cached LUT to black the moment the
	// throttle kicks in -- found via a real hardware test: a stable sun angle produced a
	// persistently black sky (SkyViewLUTNode's own plain Read<TransmittanceLUT>()/
	// Read<MultiScatteringLUT>() got zeroed every frame once the throttle stopped regenerating
	// them), while the independently-drawn sun disc kept rendering correctly.
	// detail::IsPersistentKey (RelationalCompiler.hpp) is what actually reads this.
	struct TransmittanceLUT: FixedImage2D<TransmittanceLUT, graph::Lifetime::Persistent> {
		static constexpr std::uint32_t kWidth = 256;
		static constexpr std::uint32_t kHeight = 64;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	struct MultiScatteringLUT: FixedImage2D<MultiScatteringLUT, graph::Lifetime::Persistent> {
		static constexpr std::uint32_t kWidth = 32;
		static constexpr std::uint32_t kHeight = 32;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

	struct SkyViewLUT: FixedImage2D<SkyViewLUT> {
		static constexpr std::uint32_t kWidth = 192;
		static constexpr std::uint32_t kHeight = 108;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

} // namespace brassica
