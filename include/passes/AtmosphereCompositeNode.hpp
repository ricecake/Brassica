#pragma once

#include <array>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "types/AtmosphereCompositePushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	// Fogs already-shaded scene geometry with real aerial perspective / underwater extinction --
	// distinct from SkyBackgroundNode, which only paints the empty-pixel backdrop (this node was
	// briefly, and incorrectly, named AtmosphereCompositeNode itself; see SkyBackgroundNode.hpp).
	// Runs at SubPhase::Atmosphere (900): after DeferredNode so it fogs real shaded radiance, and
	// before WaterNode (1200) -- a submerged camera looking at distant terrain needs no water mesh
	// geometry at all, just the mathematical water-level plane (shaders/atmosphere/
	// aerial_perspective.glsl's evaluateAtmosphere), so running before water costs nothing. Known
	// gap, not fixed here: a distant water *surface* itself isn't fogged by this pass since it
	// draws afterward -- water.frag calling evaluateAtmosphere on its own distToCamWater is the
	// documented follow-up.
	struct AtmosphereCompositeNode: render::NodeRegistrar<AtmosphereCompositeNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Read>,
			graph::Read<TransmittanceLUT>,
			graph::Read<MultiScatteringLUT>,
			graph::Read<SkyViewLUT>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		// No depth state: GBufferPosition is full-precision camera-relative world position
		// (TerrainNode.hpp), so composite.frag never reconstructs position from a depth buffer and
		// never binds one. enableShadingRate=false for WaterNode's reason: a translucent-looking
		// per-channel transmittance boundary must not be coarsened, and it keeps this node runnable
		// on a device without VRS.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		VertexShader                     vertShader; // shaders/atmosphere/sky.vert (shared with SkyBackgroundNode)
		FragmentShader                   fragShader; // shaders/atmosphere/composite.frag
		AtmosphereCompositePushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/atmosphere/composite.frag")) {
				spdlog::critical("AtmosphereCompositeNode shader compilation failed.");
				throw std::runtime_error("AtmosphereCompositeNode shader compilation failed.");
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
			push.sunDir = p.sunDir;
			push.sunRadiance = p.sunRadiance;
			push.moonDir = p.moonDir;
			push.moonRadiance = p.moonRadiance;
			push.multiScatScale = p.multiScatScale;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
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
					.key = graph::IdOf<MultiScatteringLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(32, 32, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<SkyViewLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.multiScatteringIndex = ctx.Index<MultiScatteringLUT>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(AtmosphereCompositePushConstants)}
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
				sizeof(AtmosphereCompositePushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(AtmosphereCompositeNode);

} // namespace brassica
