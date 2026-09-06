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
		glm::uvec4 gridParams{7, 16, 1792, 0};          // x = numLODs, y = meshletsPerRow, z = totalMeshlets
		glm::uvec4 lodOffsets0_3{0u};                   // Toroidal offsets for LOD 0-3
		glm::uvec4 lodOffsets4_7{0u};                   // Toroidal offsets for LOD 4-7
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

		vk::AccelerationStructureKHR GetTLAS() const { return tlas; }

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

		struct BufferResource {
			vk::Buffer    buffer{nullptr};
			VmaAllocation allocation{VK_NULL_HANDLE};
			vk::DeviceAddress deviceAddress{0};
		};

		VmaAllocator    lastAllocator{VK_NULL_HANDLE};
		vk::Extent2D    currentExtent{0, 0};
		TextureResource posTex;
		TextureResource normTex;
		TextureResource albTex;
		TextureResource depthTex;

		// Acceleration structure resources
		BufferResource        aabbBuffer;
		BufferResource        blasBuffer;
		vk::AccelerationStructureKHR blas{nullptr};
		BufferResource        instanceBuffer;
		BufferResource        tlasBuffer;
		vk::AccelerationStructureKHR tlas{nullptr};
		BufferResource        scratchBuffer;
		glm::vec3             lastASCameraPos{1e9f, 1e9f, 1e9f};

		void CreateGBufferTextures(vk::Extent2D extent, VmaAllocator allocator);
		void DestroyGBufferTextures(VmaAllocator allocator);
		void BuildOrUpdateAccelerationStructure(VmaAllocator allocator, const glm::vec3& cameraPos, float baseTexelSize, uint32_t numLODs);
		void DestroyAccelerationStructures();
		void InitPipelineCustom(
			vk::Instance            instance,
			vk::Device              dev,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher
		);
	};

} // namespace brassica
