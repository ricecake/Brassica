#pragma once

#include <functional>
#include <vector>

#include "passes/Pass.hpp"
#include "Shader.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	class ShaderWatcher;

	class RenderPass: public Pass {
	public:
		RenderPass(
			std::string name,
			vk::Device  device,
			vk::Format  colorFormat,
			vk::Format  depthFormat = vk::Format::eUndefined
		);
		RenderPass(
			std::string                 name,
			vk::Device                  device,
			std::span<const vk::Format> colorFormats,
			vk::Format                  depthFormat = vk::Format::eUndefined
		);
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
			vk::CompareOp                            depthCompareOp = vk::CompareOp::eLess,
			vk::CullModeFlags                        cullMode = vk::CullModeFlagBits::eBack,
			vk::PipelineCache                        pipelineCache = nullptr
		);

		void InitRenderPipeline(
			vk::Format                               colorFormat,
			vk::Format                               depthFormat = vk::Format::eUndefined,
			std::span<const vk::DescriptorSetLayout> setLayouts = {},
			std::span<const vk::PushConstantRange>   pushConstants = {},
			ShaderWatcher*                           watcher = nullptr,
			bool                                     enableDepthTest = false,
			bool                                     enableDepthWrite = false,
			vk::CompareOp                            depthCompareOp = vk::CompareOp::eLess,
			vk::CullModeFlags                        cullMode = vk::CullModeFlagBits::eBack,
			vk::PipelineCache                        pipelineCache = nullptr
		);

		// Binds the pipeline and sets the dynamic viewport/scissor for a full-frame draw. Does
		// not begin/end rendering -- that is PhysicalExecutionBackend's job now
		// (graph/PhysicalExecutionBackend.hpp's DynamicRenderingWrapper begins/ends generically,
		// driven by a node's Recipe realizations, before/after the node's own Execute() runs).
		// Public rather than protected, unlike the BeginRendering/EndRendering this replaces:
		// callers are now free-standing graph Node structs (GradientNode, TerrainNode, ...), not
		// RenderPass subclasses, so they need it from outside the class hierarchy.
		void BindForDraw(vk::CommandBuffer cmd, vk::Extent2D extent) const;

		// MDI / AZDO execution helpers
		void DrawMeshTasksIndirectEXT(
			vk::CommandBuffer            cmd,
			vk::Buffer                   buffer,
			vk::DeviceSize               offset,
			uint32_t                     drawCount,
			uint32_t                     stride,
			const DispatchLoaderDynamic& dls
		) const;

		// Direct (non-indirect) counterpart, needed by TerrainNode's mesh-shader draw.
		void DrawMeshTasksEXT(
			vk::CommandBuffer            cmd,
			uint32_t                     groupCountX,
			uint32_t                     groupCountY,
			uint32_t                     groupCountZ,
			const DispatchLoaderDynamic& dls
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
		vk::CullModeFlags       cullMode{vk::CullModeFlagBits::eBack};
		vk::PipelineCache       pipelineCache{nullptr};

		std::vector<vk::DescriptorSetLayout> storedSetLayouts;
		std::vector<vk::PushConstantRange>   storedPushConstants;

		GraphicsShader* vertOrMeshShader{nullptr};
		GraphicsShader* fragShader{nullptr};
	};

} // namespace brassica
