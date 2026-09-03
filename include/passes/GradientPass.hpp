#pragma once

#include "vulkan/vulkan.hpp"

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
		GradientPass(vk::Device device, vk::Format colorFormat);
		~GradientPass();

		void InitPipeline(vk::Device device, vk::Format colorFormat);
		void DestroyPipeline(vk::Device device);

		void RegisterPass(FrameGraph& fg, FrameGraphResource swapchainImageResource, vk::Extent2D extent);

	private:
		vk::Pipeline       pipeline{nullptr};
		vk::PipelineLayout pipelineLayout{nullptr};

		VertexShader   vertShader;
		FragmentShader fragShader;
	};

} // namespace brassica
