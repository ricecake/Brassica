#pragma once

#include "fg/FrameGraph.hpp"
#include "passes/RenderResources.hpp"
#include "vulkan/vulkan.h"

namespace brassica {

	class Engine;

	struct GradientPassData {
		FrameGraphResource target;
	};

	class GradientPass {
	public:
		GradientPass(VkDevice device, VkFormat colorFormat);
		~GradientPass();

		void InitPipeline(VkDevice device, VkFormat colorFormat);
		void DestroyPipeline(VkDevice device);

		void RegisterPass(FrameGraph& fg, FrameGraphResource swapchainImageResource, VkExtent2D extent);

	private:
		VkPipeline       pipeline{VK_NULL_HANDLE};
		VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
	};

} // namespace brassica
