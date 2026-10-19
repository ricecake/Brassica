#pragma once

#include <array>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/AtmosphereCompositePushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	// Forward-declared only: Before<VolumetricRenderNode>() below needs just the type's name
	// (TypeName<T>(), ResourceKey.hpp), never a complete type -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct VolumetricRenderNode;

	// Fogs already-shaded scene geometry with real aerial perspective / underwater extinction --
	// distinct from SkyBackgroundNode, which only paints the empty-pixel backdrop (this node was
	// briefly, and incorrectly, named AtmosphereCompositeNode itself; see SkyBackgroundNode.hpp).
	// S4: no kPhase -- runs after DeferredNode (Create-before-accessor on its sole Create<HdrColor>)
	// and before WaterNode (the explicit Before<VolumetricRenderNode>()/VolumetricRenderNode's own
	// Before<WaterNode>() chain below) -- a submerged camera looking at distant terrain needs no
	// water mesh geometry at all, just the mathematical water-level plane (shaders/atmosphere/
	// aerial_perspective.glsl's evaluateAtmosphere), so running before water costs nothing. Known
	// gap, not fixed here: a distant water *surface* itself isn't fogged by this pass since it
	// draws afterward -- water.frag calling evaluateAtmosphere on its own distToCamWater is the
	// documented follow-up.
	struct AtmosphereCompositeNode: render::NodeRegistrar<AtmosphereCompositeNode> {
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
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<HdrColor>();
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

		// No Read<GBufferNormal/Material/Depth>() -- the removed Resources claimed the whole
		// GBuffer<Read> group, but Execute below only ever indexes Position and Albedo (lesson
		// #2's own documented example of this exact over-claim). Before<VolumetricRenderNode>()
		// reproduces the schedule order S0 recorded for these two HdrColor writers.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<TransmittanceLUT>();
			r.Read<MultiScatteringLUT>();
			r.Read<SkyViewLUT>();
			r.Modify<HdrColor>();
			r.Before<VolumetricRenderNode>();
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
