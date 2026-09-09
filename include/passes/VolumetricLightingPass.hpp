#pragma once

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/Pass.hpp"
#include "passes/PassResource.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	struct VolumetricLightingData {
		FrameGraphResource injectionGrid;  // 3D Texture 160x90x64
		FrameGraphResource integratedGrid; // 3D Texture 160x90x64
	};

	struct VolumetricPushConstants {
		glm::mat4  invViewProj{1.0f};
		glm::vec4  cameraPos{0.0f, 10.0f, 20.0f, 0.5f}; // xyz = cam pos, w = baseTexelSize
		glm::vec4  sunDir{0.5f, 0.2f, 0.5f, 1.0f};      // xyz = dir to sun, w = sun intensity
		glm::vec4  sunColor{2.5f, 2.3f, 2.0f, 1.0f};    // rgb = sun color/radiance
		glm::vec4  gridDimensions{160.0f, 90.0f, 64.0f, 0.0f}; // x, y, z grid size
		glm::vec4  clipParams{0.1f, 32768.0f, 0.0f, 0.0f}; // x = near, y = far
		glm::uvec4 gridParams{8, 16, 2048, 1088};       // terrain lods, meshletsPerRow, etc.
		glm::uvec4 lodOffsets0_3{0u};                   // Toroidal offsets for LOD 0-3
		glm::uvec4 lodOffsets4_7{0u};                   // Toroidal offsets for LOD 4-7
	};

	class VolumetricLightingPass: public Pass {
	public:
		VolumetricLightingPass(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~VolumetricLightingPass() override;

		void InitPipeline(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		VolumetricLightingData RegisterPass(
			FrameGraph&                  fg,
			FrameGraphBlackboard&        blackboard,
			vk::DescriptorSet            globalDescriptorSet,
			uint32_t                     activeFrame,
			vk::ImageView                transmittanceLUTView,
			vk::Sampler                  transmittanceLUTSampler,
			vk::ImageView                clipmapImageView,
			vk::Sampler                  clipmapSampler,
			vk::AccelerationStructureKHR tlas,
			vk::Buffer                   aabbBuffer,
			const VolumetricPushConstants& pushConstants,
			VmaAllocator                 allocator = VK_NULL_HANDLE
		);

		void DestroyPipeline();

		vk::ImageView GetIntegratedImageView() const { return integratedImageView; }
		vk::ImageView GetInjectionImageView() const { return injectionImageView; }

	private:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		ComputeShader injectionShader;
		ComputeShader integrationShader;

		vk::DescriptorSetLayout injectionSetLayout{nullptr};
		vk::DescriptorSetLayout integrationSetLayout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};

		vk::DescriptorSet injectionDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet integrationDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		vk::PipelineLayout injectionPipelineLayout{nullptr};
		vk::PipelineLayout integrationPipelineLayout{nullptr};

		vk::Pipeline injectionPipeline{nullptr};
		vk::Pipeline integrationPipeline{nullptr};

		vk::Sampler sampler{nullptr};

		vk::Image     injectionImage{nullptr};
		vk::ImageView injectionImageView{nullptr};
		VmaAllocation injectionAllocation{VK_NULL_HANDLE};

		vk::Image     integratedImage{nullptr};
		vk::ImageView integratedImageView{nullptr};
		VmaAllocation integratedAllocation{VK_NULL_HANDLE};

		vk::Buffer    dummyBuffer{nullptr};
		VmaAllocation dummyAllocation{VK_NULL_HANDLE};

		vk::PipelineCache pipelineCache{nullptr};
		VmaAllocator      lastAllocator{VK_NULL_HANDLE};

		struct TextureResourceInternal {
			vk::Image     image{nullptr};
			vk::ImageView imageView{nullptr};
			VmaAllocation allocation{VK_NULL_HANDLE};
		};
		TextureResourceInternal dummy2DTex;

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
		void Create3DGridResources(VmaAllocator allocator);
		void Destroy3DGridResources();
	};

} // namespace brassica
