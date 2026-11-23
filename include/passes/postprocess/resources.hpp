#pragma once

// Postprocess-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). No logic changes from that split;
// retrofitted onto the declarative CRTP bases (resources/ResourceTypes.hpp) in a later pass --
// see passes/clouds/resources.hpp's identical comment for why.

#include "vulkan/vulkan.hpp"

#include "resources/ResourceTypes.hpp"

namespace brassica {

	namespace detail {
		// Not a base itself -- every one of TonemapComputeNode's bloom/local-tonemap mip-chain
		// keys below shares this exact format/usage, differing only in kMipIndex (ScreenMipImage2D
		// itself, resources/ResourceTypes.hpp).
		struct TonemapMipShape {
			static constexpr vk::Format kFormat = vk::Format::eR16G16B16A16Sfloat;
			static constexpr vk::ImageUsageFlags kUsage =
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		};
	} // namespace detail

	struct BloomTextureMip0: ScreenMipImage2D<BloomTextureMip0>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct BloomTextureMip1: ScreenMipImage2D<BloomTextureMip1>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct BloomTextureMip2: ScreenMipImage2D<BloomTextureMip2>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct BloomTextureMip3: ScreenMipImage2D<BloomTextureMip3>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct BloomTextureMip4: ScreenMipImage2D<BloomTextureMip4>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmExpTextureMip0: ScreenMipImage2D<LtmExpTextureMip0>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmExpTextureMip1: ScreenMipImage2D<LtmExpTextureMip1>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmExpTextureMip2: ScreenMipImage2D<LtmExpTextureMip2>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmExpTextureMip3: ScreenMipImage2D<LtmExpTextureMip3>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmExpTextureMip4: ScreenMipImage2D<LtmExpTextureMip4>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmWgtTextureMip0: ScreenMipImage2D<LtmWgtTextureMip0>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmWgtTextureMip1: ScreenMipImage2D<LtmWgtTextureMip1>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmWgtTextureMip2: ScreenMipImage2D<LtmWgtTextureMip2>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmWgtTextureMip3: ScreenMipImage2D<LtmWgtTextureMip3>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmWgtTextureMip4: ScreenMipImage2D<LtmWgtTextureMip4>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	// Same dims as mip 0 of the chain above (ltm_fuse.comp fuses down to one mip-0-sized
	// texture).
	struct LtmFusedTexture: ScreenMipImage2D<LtmFusedTexture>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	// Sky/Cloud LTM pyramid slots -- the unsuffixed Ltm*/Main keys above used to be the one
	// shared pyramid every exposure layer box-filtered into together, which let a ground texel
	// near a long horizon get pulled toward Sky's exposure ratio for several mip levels (the
	// "more sky in frame brightens the ground" bug). Giving isMain/isSky/isCloud (the 3 hard-
	// classification kinds exposureLayerWeight already special-cases everywhere else) their own
	// independent pyramid, blended only at tonemap.frag's final per-pixel step, closes that --
	// same separation the per-layer histogram/avgLuma path already had. Any other generic depth-
	// ranged custom layer still falls back to the Main slot above, same as before.
	struct LtmExpTextureMip0Sky: ScreenMipImage2D<LtmExpTextureMip0Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmExpTextureMip1Sky: ScreenMipImage2D<LtmExpTextureMip1Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmExpTextureMip2Sky: ScreenMipImage2D<LtmExpTextureMip2Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmExpTextureMip3Sky: ScreenMipImage2D<LtmExpTextureMip3Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmExpTextureMip4Sky: ScreenMipImage2D<LtmExpTextureMip4Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmWgtTextureMip0Sky: ScreenMipImage2D<LtmWgtTextureMip0Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmWgtTextureMip1Sky: ScreenMipImage2D<LtmWgtTextureMip1Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmWgtTextureMip2Sky: ScreenMipImage2D<LtmWgtTextureMip2Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmWgtTextureMip3Sky: ScreenMipImage2D<LtmWgtTextureMip3Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmWgtTextureMip4Sky: ScreenMipImage2D<LtmWgtTextureMip4Sky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmFusedTextureSky: ScreenMipImage2D<LtmFusedTextureSky>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmExpTextureMip0Cloud: ScreenMipImage2D<LtmExpTextureMip0Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmExpTextureMip1Cloud: ScreenMipImage2D<LtmExpTextureMip1Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmExpTextureMip2Cloud: ScreenMipImage2D<LtmExpTextureMip2Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmExpTextureMip3Cloud: ScreenMipImage2D<LtmExpTextureMip3Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmExpTextureMip4Cloud: ScreenMipImage2D<LtmExpTextureMip4Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmWgtTextureMip0Cloud: ScreenMipImage2D<LtmWgtTextureMip0Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

	struct LtmWgtTextureMip1Cloud: ScreenMipImage2D<LtmWgtTextureMip1Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 1;
	};

	struct LtmWgtTextureMip2Cloud: ScreenMipImage2D<LtmWgtTextureMip2Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 2;
	};

	struct LtmWgtTextureMip3Cloud: ScreenMipImage2D<LtmWgtTextureMip3Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 3;
	};

	struct LtmWgtTextureMip4Cloud: ScreenMipImage2D<LtmWgtTextureMip4Cloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 4;
	};

	struct LtmFusedTextureCloud: ScreenMipImage2D<LtmFusedTextureCloud>, detail::TonemapMipShape {
		static constexpr std::uint32_t kMipIndex = 0;
	};

} // namespace brassica
