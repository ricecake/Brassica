#pragma once

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"
#include "types/ubo/FrameUBO.hpp"

namespace brassica {

	class ShaderWatcher;

	struct MeshCubePassData {
		FrameGraphResource target;
	};

	class MeshCubePass : public RenderPass {
	public:
		MeshCubePass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);
		~MeshCubePass() override = default;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::Format              colorFormat,
			ShaderWatcher*          watcher = nullptr
		);

		FrameGraphResource RegisterPass(
			FrameGraph&           fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D          extent,
			vk::DescriptorSet     globalDescriptorSet
		);

	private:
		vk::DispatchLoaderDynamic dls;

		MeshShader     meshShader;
		FragmentShader fragShader;
	};

} // namespace brassica
