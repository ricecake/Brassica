#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "passes/TerrainPass.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	struct DeferredPushConstants {
		alignas(16) glm::mat4 viewProj{1.0f};
		alignas(16) glm::vec4 cameraPos{0.0f};
		alignas(16) glm::uvec4 gridParams{0};
		alignas(16) glm::uvec4 lodOffsets0_3{0};
		alignas(16) glm::uvec4 lodOffsets4_7{0};
		alignas(16) glm::vec4 sunDirAndIntensity{0.5f, 0.2f, 0.5f, 2.5f};
		alignas(16) glm::vec4 sunColor{1.0f, 0.95f, 0.9f, 1.0f};
	};

	struct DeferredPassData {
		FrameGraphResource target;
	};

	class DeferredPass: public RenderPass {
	public:
		DeferredPass(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~DeferredPass() override;

		void InitPipeline(
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		FrameGraphResource RegisterPass(
			FrameGraph&                  fg,
			FrameGraphBlackboard&        blackboard,
			vk::Extent2D                 extent,
			vk::DescriptorSet            globalDescriptorSet,
			uint32_t                     activeFrame = 0,
			vk::ImageView                clipmapImageView = nullptr,
			vk::Sampler                  clipmapSampler = nullptr,
			vk::AccelerationStructureKHR tlas = nullptr,
			const DeferredPushConstants& pushConstants = {}
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
