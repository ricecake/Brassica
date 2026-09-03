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
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);
		~MeshCubePass();

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);
		void DestroyPipeline(vk::Device device);

		FrameGraphResource RegisterPass(
			FrameGraph&        fg,
			FrameGraphResource inputResource,
			vk::Extent2D       extent,
			vk::DescriptorSet  globalDescriptorSet
		);

	private:
		vk::DispatchLoaderDynamic dls;
		vk::Pipeline              pipeline{nullptr};
		vk::PipelineLayout        pipelineLayout{nullptr};

		MeshShader     meshShader;
		FragmentShader fragShader;
	};

} // namespace brassica
