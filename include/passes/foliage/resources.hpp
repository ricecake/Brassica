#pragma once

// Foliage-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "passes/foliage/IFoliageManager.hpp"
#include "resources/ResourceTypes.hpp"

namespace brassica {

	// Per-biome foliage properties (color/height/width/density/flowerRatio/...), one texel-row
	// per Whittaker biome index (shaders/helpers/whittaker.glsl, 0-9) -- FoliageManager-owned,
	// written host-side via a PredefinedTextureNode whenever the UI changes a biome's properties
	// (FoliageManager::SyncTableNode, which keeps building and passing its own copy of this exact
	// desc to SetData() -- harmless, since Provision()'s catalog-always-wins rule means this
	// Desc() is what actually reaches ProvisionTexture regardless, and the two are identical so
	// no disagreement warning fires either). 5 texels wide x kFoliageBiomeCount rows, RGBA32F.
	//
	// ExternalImage2D (Lifetime::External, resources/ResourceTypes.hpp): this key's real
	// producer is a host upload with no per-frame regeneration concept at all (unlike a
	// throttled LUT), not a graph node that sometimes runs and sometimes doesn't. The base's own
	// `.persistent = true` is what actually protects this texture's real content
	// (ZeroInitializeInactiveReads, PhysicalExecutionBackend.hpp, exempts any physical resource
	// with persistent=true -- or, now, Lifetime::Persistent/External -- and HasDefinedContents()
	// -- see 0469e96, the real hardware bug this exact exemption fixed).
	struct FoliageBiomeTableTexture: ExternalImage2D<FoliageBiomeTableTexture> {
		static constexpr std::uint32_t kWidth = 5;
		static constexpr std::uint32_t kHeight = kFoliageBiomeCount;
		static constexpr vk::Format    kFormat = vk::Format::eR32G32B32A32Sfloat;
		static constexpr vk::ImageUsageFlags kUsage =
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	};

} // namespace brassica
