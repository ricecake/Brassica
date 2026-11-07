#pragma once

// Lighting-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	struct GTAOTexture: ScreenImage2D<GTAOTexture> {
		static constexpr vk::Format kFormat = vk::Format::eR16Sfloat;
		// eTransferDst: lets ZeroInitializeInactiveReads (PhysicalExecutionBackend.hpp, S5)
		// clear this texture via vkCmdClearColorImage when GTAO is disabled -- found via a
		// real hardware test: a GTAO-off frame hit a real "image not compatible with
		// VK_IMAGE_USAGE_TRANSFER_DST_BIT" validation warning on this exact clear, since
		// nothing before S5 ever needed to transfer-write into this texture.
		static constexpr vk::ImageUsageFlags kUsage = vk::ImageUsageFlagBits::eSampled |
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	};

	// Frame set binding 7 (shaders/lighting.glsl's LightningStrikePool) -- a persistent,
	// 100%-GPU-owned strike pool. climate_ca.comp claims a ring-buffer slot and writes a fresh
	// strike's origin/seed/lifetime when weather conditions trigger one; lightning_age.comp ages
	// every slot and writes its reserved light-slot entry; lightning.task reads it to plot/cull.
	// Imported once at Engine init (include/types/LightningStrike.hpp's LightningStrikePoolLayout
	// sizes the allocation) -- its handle never changes, same as TerrainReadbackStaging below, not
	// AutoExposureBuffer/LightsBuffer's "different real handle per FRAME_OVERLAP slot" case.
	struct LightningStrikePool: ImportedBuffer {};

} // namespace brassica
