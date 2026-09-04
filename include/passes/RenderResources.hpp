#pragma once

#include "vulkan/vulkan.hpp"
#include "fg/FrameGraphResource.hpp"
#include "passes/PassResource.hpp"

namespace brassica {

	struct SwapchainData {
		FrameGraphResource target;
	};

} // namespace brassica
