#pragma once

#include "passes/RenderPass.hpp"
#include "passes/RenderResources.hpp"
#include "Shader.hpp"
#include "fg/FrameGraph.hpp"
#include "fg/Blackboard.hpp"
#include <glm/glm.hpp>

namespace brassica {

	struct GltfRenderPushConstants {
		glm::mat4 viewProj{1.0f};
	};

	class GltfRenderPass : public RenderPass {
	public:
		GltfRenderPass(
			vk::Instance instance,
			vk::Device device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::DescriptorSetLayout modelGeometrySetLayout,
			vk::DescriptorSetLayout textureSetLayout,
			ShaderWatcher* watcher = nullptr
		);

		~GltfRenderPass() override = default;

		void InitPipeline(
			vk::Instance instance,
			vk::Device device,
			vk::DescriptorSetLayout globalSet0Layout,
			vk::DescriptorSetLayout modelGeometrySetLayout,
			vk::DescriptorSetLayout textureSetLayout,
			ShaderWatcher* watcher = nullptr
		);

		void RegisterPass(
			FrameGraph& fg,
			FrameGraphBlackboard& blackboard,
			vk::Extent2D extent,
			vk::DescriptorSet globalDescriptorSet,
			vk::DescriptorSet modelGeometrySet,
			vk::DescriptorSet textureSet,
			vk::Buffer indirectDrawBuffer,
			vk::Buffer drawCountBuffer,
			uint32_t maxDrawCount,
			const GltfRenderPushConstants& pushConstants
		);

		vk::DescriptorSetLayout GetModelSetLayout() const { return modelSetLayout; }

	private:
		vk::DispatchLoaderDynamic dls;
		TaskShader taskShader;
		MeshShader meshShader;
		FragmentShader fragShader;

		vk::DescriptorSetLayout modelSetLayout{nullptr};
	};

} // namespace brassica
