#pragma once

#include <array>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/SkyPushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	// Draws the sky-view/transmittance LUTs composited with the sun/moon/star discs into
	// AtmosphereRadiance, DeferredNode's backdrop wherever no opaque surface was rendered.
	// Deliberately not named AtmosphereCompositeNode: that name is reserved for the node that fogs
	// already-shaded scene geometry with aerial perspective/underwater extinction (see
	// AtmosphereCompositeNode.hpp) -- a distinct concept from drawing the sky background, even
	// though this node was briefly (and incorrectly) named that.
	struct SkyBackgroundNode: render::NodeRegistrar<SkyBackgroundNode> {
		using Resources =
			graph::Declares<graph::Read<SkyViewLUT>, graph::Read<TransmittanceLUT>, graph::Create<AtmosphereRadiance>>;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		SkyPushConstants         push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert");
			fragShader.CompileFragmentFromFile(services.device, "shaders/atmosphere/sky.frag");
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
			push.sunDirAndAureole = glm::vec4(p.sunDir, 0.5f);
			push.moonDirAndCirrus = glm::vec4(p.moonDir, 0.3f);
			push.sunRadianceAndSkyExp = glm::vec4(p.sunRadiance, p.skyExposure);
			push.rotToCamQuat = glm::vec4(p.rotToCam.x, p.rotToCam.y, p.rotToCam.z, p.rotToCam.w);
			push.worldScale = p.worldScale;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<SkyViewLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AtmosphereRadiance>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(SkyPushConstants)}
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

			vkCmd
				.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(SkyPushConstants), &push);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(SkyBackgroundNode);

} // namespace brassica
