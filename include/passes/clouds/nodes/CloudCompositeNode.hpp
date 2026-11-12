#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Forward-declared only: After<CirrusNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct CirrusNode;

	struct CloudCompositePushConstants {
		std::uint32_t hdrColorIndex{0};
		std::uint32_t cloudColorIndex{0};
		std::uint32_t cloudTransmittanceIndex{0};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t cloudMomentsIndex{0};
	};

	// Blends CloudTemporalNode's (Stage B) accumulated output into HdrColor -- not
	// CloudRenderNode's raw per-frame raymarch directly, now that Stage B exists. Takes over
	// CloudRenderNode's old position in the HdrColor writer chain -- it's the one Modifying
	// HdrColor now, so every Before<>/After<> ordering that existed because of shared HdrColor
	// access (CirrusNode, VolumetricRenderNode, ParticleSystemNode, LightningRenderNode) points
	// at this node instead. After<CloudTemporalNode>() isn't declared explicitly below --
	// Read<CloudAccumulatedColor/Transmittance>() already orders this after CloudTemporalNode's
	// sole Create<> of both via the usual Create-before-accessor edge.
	struct CloudCompositeNode: render::NodeRegistrar<CloudCompositeNode> {
		// depthTest/depthWrite false: this pass doesn't touch GBufferDepth's depth aspect at all
		// (its own occlusion test already runs by hand against CloudMoments/GBufferPosition, see
		// cloud_composite.frag) -- Modify<GBufferDepth>() below is only to gain stencil-write
		// access, and depthFormat on the BindGraphics call is what makes StencilMode::Write legal
		// (PipelineLibrary.cpp requires a real depthFormat to enable the stencil test at all).
		// stencilReference=2 (passed to BindGraphics) marks every non-discarded fragment as cloud
		// -- see cloud_composite.frag's discard for which fragments that excludes.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = false,
			.depthWrite = false,
			.stencilMode = render::StencilMode::Write,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*      pipelineLibrary = nullptr;
		VertexShader                  vertShader; // shaders/atmosphere/sky.vert
		FragmentShader                fragShader; // shaders/cloud_composite.frag
		CloudCompositePushConstants   push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cloud_composite.frag")) {
				spdlog::critical("CloudCompositeNode shader compilation failed.");
				throw std::runtime_error("CloudCompositeNode shader compilation failed.");
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

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<CloudAccumulatedColor>();
			r.Read<CloudAccumulatedTransmittance>();
			r.Read<CloudMoments>();
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Modify<HdrColor>();
			// ReadWrite, not Read: a genuinely read-only declaration gets storeOp=eNone for both
			// depth and stencil aspects (DynamicRenderingWrapper::Begin mirrors stencil's load/
			// store onto whatever depth got), which would silently discard this pass's own stencil
			// write. depthTest/depthWrite are both false in kPipelineState, so the real depth
			// aspect this loads is otherwise untouched -- only stencil (classification=2, see
			// cloud_composite.frag) actually changes. The opaque GBuffer pass's existing sole
			// Create<GBufferDepth>() already orders this node after it via the usual Create-
			// before-accessor edge, same as every pre-existing Read<GBufferDepth>() elsewhere
			// (TonemapNode, TonemapComputeNode) -- no new explicit ordering needed for that.
			r.Modify<GBufferDepth>();
			r.After<CirrusNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.cloudColorIndex = ctx.Index<CloudAccumulatedColor>();
			push.cloudTransmittanceIndex = ctx.Index<CloudAccumulatedTransmittance>();
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.cloudMomentsIndex = ctx.Index<CloudMoments>();

			std::array<GraphicsShader*, 2> stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			// depthFormat (GBufferDepth's real format) is what lets PipelineLibrary legally enable
			// the stencil test at all for a StencilMode other than Disabled -- stencilReference=2
			// is the classification value every non-discarded fragment stamps (cloud_composite.frag).
			render::ResolvedPipeline resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eFragment,
                GBufferDepth::kFormat,
                render::ExtraSets{},
                /*stencilReference=*/2u
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(CloudCompositeNode);

} // namespace brassica
