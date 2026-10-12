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
		alignas(16) glm::vec4 scaledCamRight{1.0f, 0.0f, 0.0f, 0.0f}; // xyz = camRight * aspect * tanHalfFov
		alignas(16) glm::vec4 scaledCamUp{0.0f, 1.0f, 0.0f, 0.0f};    // xyz = camUp * tanHalfFov
		alignas(16) glm::uvec4 cascadeScatExtIdx01{0xFFFFFFFFu};    // x,y = C0 scat/ext, z,w = C1 scat/ext
		alignas(16) glm::uvec4 cascadeScatExtIdx23{0xFFFFFFFFu};    // x,y = C2 scat/ext, z,w = C3 scat/ext
		alignas(16) glm::uvec4 gBufferConfig{0u};                    // x = gPositionIndex, y = gDepthIndex, z = hdrColorIndex, w = maxDist
	};
	static_assert(sizeof(VolumetricRenderPushConstants) == 80, "VolumetricRenderPushConstants size must be 80 bytes");

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
			graph::Read<VolumetricCascade3Scattering>,
			graph::Read<VolumetricCascade3Extinction>,
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
			float tanHalfFov = std::tan(p.fov * 0.5f);
			push.scaledCamRight = glm::vec4(p.cameraRight * p.aspectRatio * tanHalfFov, 0.0f);
			push.scaledCamUp = glm::vec4(p.cameraUp * tanHalfFov, 0.0f);
			push.gBufferConfig.w = std::bit_cast<std::uint32_t>(51200.0f);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.reserve(10);

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
					.key = graph::IdOf<VolumetricCascade3Scattering>(),
					.access = graph::AccessKind::Read,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade3Extinction>(),
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
			push.cascadeScatExtIdx01 = glm::uvec4(
				ctx.Index<VolumetricCascade0Scattering>(),
				ctx.Index<VolumetricCascade0Extinction>(),
				ctx.Index<VolumetricCascade1Scattering>(),
				ctx.Index<VolumetricCascade1Extinction>()
			);
			push.cascadeScatExtIdx23 = glm::uvec4(
				ctx.Index<VolumetricCascade2Scattering>(),
				ctx.Index<VolumetricCascade2Extinction>(),
				ctx.Index<VolumetricCascade3Scattering>(),
				ctx.Index<VolumetricCascade3Extinction>()
			);

			push.gBufferConfig.x = ctx.Index<GBufferPosition>();
			push.gBufferConfig.y = ctx.Index<GBufferDepth>();
			push.gBufferConfig.z = ctx.Index<HdrColor>();

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
