#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	struct GradientPassData {
		FrameGraphResource target;
	};

	class GradientPass : public RenderPass {
	public:
		GradientPass(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher = nullptr);
		~GradientPass() override = default;

		void InitPipeline(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher = nullptr);

		FrameGraphResource RegisterPass(FrameGraph& fg, FrameGraphBlackboard& blackboard, vk::Extent2D extent);

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;
	};

} // namespace brassica
