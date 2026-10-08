#pragma once

#include <array>
#include <cmath>
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

	struct alignas(16) VolumetricRenderPushConstants {
		alignas(16) glm::vec4 camForward{0.0f, 0.0f, -1.0f, 0.0f};
		alignas(16) glm::vec4 camUp{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 camRight{1.0f, 0.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 fovAspect{1.0f, 1.777f, 0.0f, 0.0f}; // tanHalfFov, aspect
		alignas(16) glm::uvec4 cascadeScatteringIdx{0xFFFFFFFFu};
		alignas(16) glm::uvec4 cascadeExtinctionIdx{0xFFFFFFFFu};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t hdrColorIndex{0};
		float         maxDistance{12800.0f};
	};

	// Composites froxel volumetric scattering and extinction onto HdrColor.
	// Operates in SubPhase::Atmosphere (Phase 900).
	struct VolumetricRenderNode: render::NodeRegistrar<VolumetricRenderNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Read>,
			graph::Read<VolumetricCascade0Scattering>,
			graph::Read<VolumetricCascade0Extinction>,
			graph::Read<VolumetricCascade1Scattering>,
			graph::Read<VolumetricCascade1Extinction>,
			graph::Read<VolumetricCascade2Scattering>,
			graph::Read<VolumetricCascade2Extinction>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*      pipelineLibrary = nullptr;
		VertexShader                   vertShader;
		FragmentShader                 fragShader;
		VolumetricRenderPushConstants  push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/volumetric_render.frag")) {
				spdlog::critical("VolumetricRenderNode shader compilation failed.");
				throw std::runtime_error("VolumetricRenderNode shader compilation failed.");
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
			push.camForward = glm::vec4(p.cameraForward, 0.0f);
			push.camUp = glm::vec4(p.cameraUp, 0.0f);
			push.camRight = glm::vec4(p.cameraRight, 0.0f);
			push.fovAspect = glm::vec4(std::tan(p.fov * 0.5f), p.aspectRatio, 0.0f, 0.0f);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.reserve(8);

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade0Scattering>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade0Extinction>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade1Scattering>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade1Extinction>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade2Scattering>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade2Extinction>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.cascadeScatteringIdx.x = ctx.Index<VolumetricCascade0Scattering>();
			push.cascadeScatteringIdx.y = ctx.Index<VolumetricCascade1Scattering>();
			push.cascadeScatteringIdx.z = ctx.Index<VolumetricCascade2Scattering>();

			push.cascadeExtinctionIdx.x = ctx.Index<VolumetricCascade0Extinction>();
			push.cascadeExtinctionIdx.y = ctx.Index<VolumetricCascade1Extinction>();
			push.cascadeExtinctionIdx.z = ctx.Index<VolumetricCascade2Extinction>();

			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.hdrColorIndex = ctx.Index<HdrColor>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(VolumetricRenderPushConstants)}
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
			vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(VolumetricRenderPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(VolumetricRenderNode);

} // namespace brassica
