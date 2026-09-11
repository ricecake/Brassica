#pragma once

#include <array>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

namespace brassica {

	// Trivial case for the node/pass unification: no descriptors, no push constants, a fixed
	// pair of shaders and a fixed output format. Replaces the old GradientPass -- there is no
	// persistent Pass object anymore; the node is reconstructed fresh every frame (see
	// Engine::gradientVertShader/gradientFragShader and Engine::pipelineLibrary, which are what
	// actually persist), and ResolveCached makes re-resolving the same request every frame a
	// cache hit rather than a real pipeline rebuild.
	struct GradientNode {
		using Resources = graph::Declares<graph::Create<GradientBackground>>;

		// Only cullMode needs to move off GraphicsPipelineState's own defaults -- a fullscreen
		// triangle drawn eBack (the default) would be culled depending on winding, exactly like
		// GradientPass::InitPipeline's explicit vk::CullModeFlagBits::eNone used to guard against.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader*            vertShader = nullptr;
		FragmentShader*          fragShader = nullptr;
		vk::Extent2D             extent;

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

		void Execute(graph::NodeContext& ctx) {
			std::array<GraphicsShader*, 2> stages{vertShader, fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};

			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}

			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});
			vkCmd.draw(3, 1, 0, 0);
		}
	};

} // namespace brassica
