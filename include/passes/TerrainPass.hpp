#pragma once

#include <glm/glm.hpp>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	struct TerrainPassData {
		FrameGraphResource positionTarget;
		FrameGraphResource normalTarget;
		FrameGraphResource albedoTarget;
		FrameGraphResource depthTarget;
	};

	struct TerrainPushConstants {
		glm::mat4  viewProj{1.0f};
		glm::vec4  cameraPos{0.0f, 10.0f, 20.0f, 0.5f}; // xyz = camera position, w = baseTexelSize
		glm::uvec4 gridParams{4, 16, 256, 0}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets
	};

	class TerrainPass : public RenderPass {
	public:
		TerrainPass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr
		);
		~TerrainPass() override;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr
		);

		void UpdateClipmapDescriptor(vk::ImageView clipmapImageView, vk::Sampler clipmapSampler);

		void RegisterPass(
			FrameGraph&           fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D          extent,
			vk::DescriptorSet     globalDescriptorSet,
			const TerrainPushConstants& pushConstants,
			VmaAllocator          allocator = VK_NULL_HANDLE
		);

	private:
		vk::DispatchLoaderDynamic dls;

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		vk::DescriptorSetLayout terrainSet1Layout{nullptr};
		vk::DescriptorPool      terrainDescriptorPool{nullptr};
		vk::DescriptorSet       terrainDescriptorSet{nullptr};

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
		void InitPipelineCustom(
			vk::Instance            instance,
			vk::Device              dev,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher
		);
	};

} // namespace brassica
