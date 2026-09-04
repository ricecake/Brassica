#pragma once

#include "vulkan/vulkan.hpp"
#include "fg/FrameGraphResource.hpp"
#include "passes/PassResource.hpp"

namespace brassica {

	struct SwapchainData {
		FrameGraphResource target;
	};

	struct GBufferData {
		FrameGraphResource positionTarget; // eR16G16B16A16Sfloat
		FrameGraphResource normalTarget;   // eR16G16B16A16Sfloat
		FrameGraphResource albedoTarget;   // eR8G8B8A8Unorm
		FrameGraphResource depthTarget;    // eD32Sfloat
	};

} // namespace brassica
