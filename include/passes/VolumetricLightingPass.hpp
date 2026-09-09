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
		FrameGraphResource injectionGrid;  // 3D Texture 160x90x256
		FrameGraphResource integratedGrid; // 3D Texture 160x90x256
	};

	struct VolumetricPushConstants {
		glm::mat4 invViewProj{1.0f};
		glm::mat4 prevViewProj{1.0f};
		glm::vec4 cameraPos{0.0f, 10.0f, 20.0f, 0.5f}; // xyz = cam pos, w = baseTexelSize
		glm::vec4 sunDir{0.5f, 0.2f, 0.5f, 1.0f};       // xyz = dir to sun, w = sun intensity
		glm::vec4 sunColor{2.5f, 2.3f, 2.0f, 1.0f};     // rgb = sun color
		glm::vec4 params0{0.8f, 1.0f, 1.0f, 1.0f};      // x = anisotropy (g), y = intensity, z = ambientScale, w = shadowSensitivity
		glm::vec4 params1{2.0f, 0.95f, 1.0f, 1.0f};     // x = lightAccent, y = temporalAlpha, z = rayleighScale, w = mieScale
		glm::vec4 cascadeDistances{20.0f, 60.0f, 200.0f, 1000.0f}; // 4 cascades
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

		// Double-buffered 3D history textures for temporal reprojection
		vk::Image     history3DImages[2]{nullptr, nullptr};
		vk::ImageView history3DViews[2]{nullptr, nullptr};
		VmaAllocation history3DAllocations[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
		uint32_t      historyIndex3D{0};
		bool          hasHistory3D{false};

		vk::Buffer    dummyBuffer{nullptr};
		VmaAllocation dummyAllocation{VK_NULL_HANDLE};

		struct TextureResourceInternal {
			vk::Image     image{nullptr};
			vk::ImageView imageView{nullptr};
			VmaAllocation allocation{VK_NULL_HANDLE};
		};
		TextureResourceInternal dummy2DTex;

		vk::PipelineCache pipelineCache{nullptr};
		VmaAllocator      lastAllocator{VK_NULL_HANDLE};

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
		void Create3DGridResources(VmaAllocator allocator);
		void Destroy3DGridResources();
	};

} // namespace brassica
