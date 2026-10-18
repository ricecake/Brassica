#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct alignas(16) CloudTaaPushConstants {
		std::uint32_t cloudRenderIndex{0};
		std::uint32_t cloudDepthIndex{0};
		std::uint32_t cloudVelocityIndex{0};
		std::uint32_t historyColorIndex{0};
		std::uint32_t historyMomentsIndex{0};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t frameIndex{0};
		float         blendAlpha{0.0625f};
	};

	// Volumetric TAA pass for cloud rendering. Reconstructs full-resolution clouds from 1/16th resolution
	// intermediate targets using temporal reprojection, 3x3 neighborhood clamping, variance-based moment tracking,
	// and composites anti-aliased clouds into HdrColor.
	struct CloudTaaNode: render::NodeRegistrar<CloudTaaNode> {
		using Resources = graph::Declares<
			graph::Read<CloudRenderTexture>,
			graph::Read<CloudDepthTexture>,
			graph::Read<CloudVelocityTexture>,
			graph::Read<graph::History<CloudHistoryTexture>>,
			graph::Read<graph::History<CloudHistoryMomentsTexture>>,
			graph::Create<CloudHistoryTexture>,
			graph::Create<CloudHistoryMomentsTexture>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = graph::Phase(1260);

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader; // shaders/atmosphere/sky.vert
		FragmentShader           fragShader; // shaders/cloud_taa.frag
		CloudTaaPushConstants    push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cloud_taa.frag")) {
				spdlog::critical("CloudTaaNode shader compilation failed.");
				throw std::runtime_error("CloudTaaNode shader compilation failed.");
			}
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

		void SetFrameParams(const render::NodeFrameParams& /*p*/) {}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.reserve(8);
			std::uint32_t quarterWidth = (ctx.width + 3) / 4;
			std::uint32_t quarterHeight = (ctx.height + 3) / 4;

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudHistoryTexture>(),
					.access = graph::AccessKind::Write,
					.desc = CloudHistoryTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudHistoryMomentsTexture>(),
					.access = graph::AccessKind::Write,
					.desc = CloudHistoryMomentsTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudRenderTexture>(),
					.access = graph::AccessKind::Read,
					.desc = CloudRenderTextureDesc(quarterWidth, quarterHeight),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudDepthTexture>(),
					.access = graph::AccessKind::Read,
					.desc = CloudDepthTextureDesc(quarterWidth, quarterHeight),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVelocityTexture>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVelocityTextureDesc(quarterWidth, quarterHeight),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<graph::History<CloudHistoryTexture>>(),
					.access = graph::AccessKind::Read,
					.desc = CloudHistoryTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<graph::History<CloudHistoryMomentsTexture>>(),
					.access = graph::AccessKind::Read,
					.desc = CloudHistoryMomentsTextureDesc(ctx.width, ctx.height),
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.cloudRenderIndex = ctx.Index<CloudRenderTexture>();
			push.cloudDepthIndex = ctx.Index<CloudDepthTexture>();
			push.cloudVelocityIndex = ctx.Index<CloudVelocityTexture>();
			push.historyColorIndex = ctx.Index<graph::History<CloudHistoryTexture>>();
			push.historyMomentsIndex = ctx.Index<graph::History<CloudHistoryMomentsTexture>>();
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.frameIndex = static_cast<std::uint32_t>(ctx.frameIndex);

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 3>              colorFormats{
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR16G16B16A16Sfloat
			};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(CloudTaaPushConstants)}
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
				sizeof(CloudTaaPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(CloudTaaNode);

} // namespace brassica
