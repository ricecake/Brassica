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

} // namespace brassica
