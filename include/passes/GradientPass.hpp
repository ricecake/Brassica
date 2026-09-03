#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"

namespace brassica {

	class Engine;
	class ShaderWatcher;

	struct GradientPassData {
		FrameGraphResource target;
	};

	class GradientPass {
	public:
		GradientPass(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher = nullptr);
		~GradientPass();

		void InitPipeline(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher = nullptr);
		void DestroyPipeline(vk::Device device);

		FrameGraphResource RegisterPass(FrameGraph& fg, FrameGraphBlackboard& blackboard, vk::Extent2D extent);

	private:
		vk::Pipeline       pipeline{nullptr};
		vk::PipelineLayout pipelineLayout{nullptr};

		VertexShader   vertShader;
		FragmentShader fragShader;
	};

} // namespace brassica
