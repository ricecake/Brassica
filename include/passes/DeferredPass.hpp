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
			vk::Buffer                   aabbBuffer = nullptr,
			vk::ImageView                volumetricIntegratedView = nullptr,
			const TerrainPushConstants&  pushConstants = {},
			VmaAllocator                 allocator = VK_NULL_HANDLE
		);

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;

		static constexpr uint32_t FRAME_OVERLAP = 2;
		vk::DescriptorSetLayout   gbufferSetLayout{nullptr};
		vk::DescriptorPool        descriptorPool{nullptr};
		vk::DescriptorSet         gbufferDescriptorSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::Sampler               sampler{nullptr};

		vk::Buffer                dummyBuffer{nullptr};
		VmaAllocation             dummyAllocation{VK_NULL_HANDLE};
		vk::Image                 dummy3DImage{nullptr};
		vk::ImageView             dummy3DView{nullptr};
		VmaAllocation             dummy3DAllocation{VK_NULL_HANDLE};
		VmaAllocator              lastAllocator{VK_NULL_HANDLE};

		void CreateDescriptorResources(vk::Device device);
		void CleanupDescriptorResources();
		void EnsureDummyBuffer(VmaAllocator allocator);
	};

} // namespace brassica
