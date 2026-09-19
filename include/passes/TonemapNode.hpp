#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

#include "VulkanCompat.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/TonemapPushConstants.hpp"

namespace brassica {

	struct TonemapNode: render::NodeRegistrar<TonemapNode> {
		using Resources = graph::Declares<graph::Read<HdrColor>, graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = brassica::SubPhase::ToneMapping;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		vk::Format               swapchainFormat = vk::Format::eUndefined;
		TonemapPushConstants     push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			swapchainFormat = services.swapchainFormat;
			vertShader.CompileVertexFromFile(services.device, "shaders/tonemap.vert");
			fragShader.CompileFragmentFromFile(services.device, "shaders/tonemap.frag");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&vertShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			vertShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& /*p*/) {
			push = s_tonemapPush;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push = s_tonemapPush;
			push.hdrColorIndex = ctx.Index<HdrColor>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(TonemapPushConstants)}
			};
			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, boundSets, nullptr);
			}

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(TonemapPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(TonemapNode);

} // namespace brassica
