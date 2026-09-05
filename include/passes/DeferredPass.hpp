#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	struct DeferredPassData {
		FrameGraphResource target;
	};

	class DeferredPass : public RenderPass {
	public:
		DeferredPass(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);
		~DeferredPass() override;

		void InitPipeline(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);

		FrameGraphResource RegisterPass(
			FrameGraph&           fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D          extent,
			vk::DescriptorSet     globalDescriptorSet,
			uint32_t              activeFrame = 0
		);

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;

		static constexpr uint32_t FRAME_OVERLAP = 2;
		vk::DescriptorSetLayout   gbufferSetLayout{nullptr};
		vk::DescriptorPool        descriptorPool{nullptr};
		vk::DescriptorSet         gbufferDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::Sampler               sampler{nullptr};

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
	};

} // namespace brassica
