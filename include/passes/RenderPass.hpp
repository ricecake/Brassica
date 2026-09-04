#pragma once

#include <functional>
#include <vector>
#include "passes/Pass.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	class RenderPass : public Pass {
	public:
		RenderPass(std::string name, vk::Device device, vk::Format colorFormat, vk::Format depthFormat = vk::Format::eUndefined);
		~RenderPass() override;

		void SetShaders(GraphicsShader* vertexOrMesh, GraphicsShader* fragment);

		void InitRenderPipeline(
			vk::Format                        colorFormat,
			vk::Format                        depthFormat = vk::Format::eUndefined,
			std::span<const vk::DescriptorSetLayout> setLayouts = {},
			std::span<const vk::PushConstantRange>   pushConstants = {},
			ShaderWatcher*                    watcher = nullptr
		);

	protected:
		vk::Format colorFormat{vk::Format::eUndefined};
		vk::Format depthFormat{vk::Format::eUndefined};

		std::vector<vk::DescriptorSetLayout> storedSetLayouts;
		std::vector<vk::PushConstantRange>   storedPushConstants;

		GraphicsShader* vertOrMeshShader{nullptr};
		GraphicsShader* fragShader{nullptr};

		void BeginRendering(
			vk::CommandBuffer        cmd,
			vk::Extent2D             extent,
			std::span<const vk::RenderingAttachmentInfo> colorAttachments,
			const vk::RenderingAttachmentInfo*           depthAttachment = nullptr
		) const;

		void EndRendering(vk::CommandBuffer cmd) const;
	};

} // namespace brassica
