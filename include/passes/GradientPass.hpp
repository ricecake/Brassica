#pragma once

#include "vulkan/vulkan.h"

#include "fg/FrameGraph.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"

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

		VertexShader   vertShader;
		FragmentShader fragShader;
	};

} // namespace brassica
