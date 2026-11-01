#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
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
	};

	// Blends CloudRenderNode's (now compute) output into HdrColor. Takes over CloudRenderNode's
	// old position in the HdrColor writer chain -- it's the one Modifying HdrColor now, so every
	// Before<CloudRenderNode>()/After<CloudRenderNode>() ordering that existed because of shared
	// HdrColor access (CirrusNode, VolumetricRenderNode, ParticleSystemNode, LightningRenderNode)
	// now points at this node instead. After<CloudRenderNode>() itself isn't declared explicitly
	// below -- Read<CloudColor/CloudTransmittance>() already orders this after CloudRenderNode's
	// sole Create<> of both via the usual Create-before-accessor edge.
	struct CloudCompositeNode: render::NodeRegistrar<CloudCompositeNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
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
			r.Read<CloudColor>();
			r.Read<CloudTransmittance>();
			r.Modify<HdrColor>();
			r.After<CirrusNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.cloudColorIndex = ctx.Index<CloudColor>();
			push.cloudTransmittanceIndex = ctx.Index<CloudTransmittance>();

			std::array<GraphicsShader*, 2> stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			render::ResolvedPipeline       resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eFragment
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
