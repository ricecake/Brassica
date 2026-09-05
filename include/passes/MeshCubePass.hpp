#pragma once

#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"

namespace brassica {

	class ShaderWatcher;

	struct MeshCubePassData {
		FrameGraphResource positionTarget;
		FrameGraphResource normalTarget;
		FrameGraphResource albedoTarget;
		FrameGraphResource depthTarget;
	};

	class MeshCubePass : public RenderPass {
	public:
		MeshCubePass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr
		);
		~MeshCubePass() override;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr
		);

		void RegisterPass(
			FrameGraph&           fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D          extent,
			vk::DescriptorSet     globalDescriptorSet,
			VmaAllocator          allocator = VK_NULL_HANDLE
		);

	private:
		vk::DispatchLoaderDynamic dls;

		MeshShader     meshShader;
		FragmentShader fragShader;

		struct TextureResource {
			vk::Image     image{nullptr};
			vk::ImageView imageView{nullptr};
			VmaAllocation allocation{VK_NULL_HANDLE};
		};

		VmaAllocator    lastAllocator{VK_NULL_HANDLE};
		vk::Extent2D    currentExtent{0, 0};
		TextureResource posTex;
		TextureResource normTex;
		TextureResource albTex;
		TextureResource depthTex;

		void CreateGBufferTextures(vk::Extent2D extent, VmaAllocator allocator);
		void DestroyGBufferTextures(VmaAllocator allocator);
	};

} // namespace brassica
