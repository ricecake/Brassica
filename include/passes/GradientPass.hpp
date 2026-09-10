#pragma once

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPass.hpp"
#include "passes/ResourceKeys.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	class GradientPass: public RenderPass {
	public:
		GradientPass(
			vk::Device        device,
			vk::Format        colorFormat,
			ShaderWatcher*    watcher = nullptr,
			vk::PipelineCache pCache = nullptr
		);
		~GradientPass() override;

		void InitPipeline(
			vk::Device        device,
			vk::Format        colorFormat,
			ShaderWatcher*    watcher = nullptr,
			vk::PipelineCache pCache = nullptr
		);

	private:
		VertexShader   vertShader;
		FragmentShader fragShader;
	};

	// Registry-managed replacement for the old manual bgImage/bgImageView/currentExtent
	// reallocation-on-resize logic in RegisterPass -- the registry now owns that texture's
	// lifetime and reuses it across frames automatically whenever its desc doesn't change
	// (PhysicalResourceRegistry::ProvisionTexture's desc-match early return).
	struct GradientNode {
		using Resources = graph::Declares<graph::Create<GradientBackground>>;

		GradientPass* pass;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GradientBackground>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
			pass->BindForDraw(vkCmd, extent);
			vkCmd.draw(3, 1, 0, 0);
		}

		vk::Extent2D extent;
	};

} // namespace brassica
