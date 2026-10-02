#pragma once

#include <array>
#include <cstdint>

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
#include "spdlog/spdlog.h"

namespace brassica {

	struct CloudRenderPushConstants {
		alignas(16) glm::uvec4 cascadeSampledIdx{0};
		alignas(16) glm::vec4 cameraPos{0.0f};
	};
	static_assert(sizeof(CloudRenderPushConstants) == 32, "CloudRenderPushConstants size must be 32 bytes");

	// Reads the 3 cloud volume cascade textures and renders them to visible clouds in HdrColor.
	// Minimally populated graphics fragment pass for cloud volume sampling and composition.
	struct CloudRenderNode: render::NodeRegistrar<CloudRenderNode> {
		using Resources = graph::Declares<
			graph::Read<CloudVolumeCascade0>,
			graph::Read<CloudVolumeCascade1>,
			graph::Read<CloudVolumeCascade2>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader; // shaders/atmosphere/sky.vert
		FragmentShader           fragShader; // shaders/cloud_render.frag
		CloudRenderPushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cloud_render.frag")) {
				spdlog::critical("CloudRenderNode shader compilation failed.");
				throw std::runtime_error("CloudRenderNode shader compilation failed.");
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

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.cameraPos = glm::vec4(p.cameraPosition, 1.0f);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.reserve(4);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade0>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade1>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade2>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.cascadeSampledIdx.x = ctx.Index<CloudVolumeCascade0>();
			push.cascadeSampledIdx.y = ctx.Index<CloudVolumeCascade1>();
			push.cascadeSampledIdx.z = ctx.Index<CloudVolumeCascade2>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(CloudRenderPushConstants)}
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
				sizeof(CloudRenderPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(CloudRenderNode);

} // namespace brassica
