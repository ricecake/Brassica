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
		RenderPass(std::string name, vk::Device device, std::span<const vk::Format> colorFormats, vk::Format depthFormat = vk::Format::eUndefined);
		~RenderPass() override;

		void SetShaders(GraphicsShader* vertexOrMesh, GraphicsShader* fragment);

		void InitRenderPipeline(
			std::span<const vk::Format>              colorFormats,
			vk::Format                               depthFormat = vk::Format::eUndefined,
			std::span<const vk::DescriptorSetLayout> setLayouts = {},
			std::span<const vk::PushConstantRange>   pushConstants = {},
			ShaderWatcher*                           watcher = nullptr,
			bool                                     enableDepthTest = false,
			bool                                     enableDepthWrite = false,
			vk::CompareOp                            depthCompareOp = vk::CompareOp::eLess
		);

		void InitRenderPipeline(
			vk::Format                               colorFormat,
			vk::Format                               depthFormat = vk::Format::eUndefined,
			std::span<const vk::DescriptorSetLayout> setLayouts = {},
			std::span<const vk::PushConstantRange>   pushConstants = {},
			ShaderWatcher*                           watcher = nullptr,
			bool                                     enableDepthTest = false,
			bool                                     enableDepthWrite = false,
			vk::CompareOp                            depthCompareOp = vk::CompareOp::eLess
		);

		// MDI / AZDO execution helpers
		void DrawMeshTasksIndirectEXT(
			vk::CommandBuffer                cmd,
			vk::Buffer                       buffer,
			vk::DeviceSize                   offset,
			uint32_t                         drawCount,
			uint32_t                         stride,
			const vk::DispatchLoaderDynamic& dls
		) const;

		void DrawIndexedIndirect(
			vk::CommandBuffer cmd,
			vk::Buffer        buffer,
			vk::DeviceSize    offset,
			uint32_t          drawCount,
			uint32_t          stride
		) const;

		void DrawIndirect(
			vk::CommandBuffer cmd,
			vk::Buffer        buffer,
			vk::DeviceSize    offset,
			uint32_t          drawCount,
			uint32_t          stride
		) const;

	protected:
		std::vector<vk::Format> colorFormats;
		vk::Format              depthFormat{vk::Format::eUndefined};
		bool                    depthTestEnable{false};
		bool                    depthWriteEnable{false};
		vk::CompareOp           depthCompareOp{vk::CompareOp::eLess};

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
