#pragma once

#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

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
		~GradientPass() override;

		void InitPipeline(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher = nullptr);

		FrameGraphResource RegisterPass(
			FrameGraph&           fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D          extent,
			VmaAllocator          allocator = VK_NULL_HANDLE
		);

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;

		VmaAllocator  lastAllocator{VK_NULL_HANDLE};
		vk::Extent2D  currentExtent{0, 0};
		vk::Image     bgImage{nullptr};
		vk::ImageView bgImageView{nullptr};
		VmaAllocation bgAllocation{VK_NULL_HANDLE};

		void DestroyTextureResource(VmaAllocator allocator);
	};

} // namespace brassica
