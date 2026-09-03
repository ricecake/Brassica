#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/FrameGraph.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"

namespace brassica {

	class ShaderWatcher;

	struct MeshCubePassData {
		FrameGraphResource target;
	};

	class MeshCubePass {
	public:
		MeshCubePass(
			vk::Instance       instance,
			vk::PhysicalDevice physicalDevice,
			vk::Device         device,
			vk::Format         colorFormat,
			ShaderWatcher*     watcher = nullptr
		);
		~MeshCubePass();

		void InitPipeline(
			vk::Instance       instance,
			vk::PhysicalDevice physicalDevice,
			vk::Device         device,
			vk::Format         colorFormat,
			ShaderWatcher*     watcher = nullptr
		);
		void DestroyPipeline(vk::Device device);

		void RegisterPass(
			FrameGraph&        fg,
			FrameGraphResource swapchainImageResource,
			vk::Extent2D       extent,
			const FrameUBO&    uboData,
			uint32_t           frameIndex
		);

	private:
		vk::DispatchLoaderDynamic dls;
		vk::Pipeline            pipeline{nullptr};
		vk::PipelineLayout      pipelineLayout{nullptr};
		vk::DescriptorSetLayout descriptorSetLayout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};

		static constexpr unsigned int FRAME_OVERLAP = 2;
		vk::Buffer            uboBuffers[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DeviceMemory      uboMemory[FRAME_OVERLAP]{nullptr, nullptr};
		void*                 uboMapped[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet     descriptorSets[FRAME_OVERLAP]{nullptr, nullptr};

		MeshShader     meshShader;
		FragmentShader fragShader;
	};

} // namespace brassica
