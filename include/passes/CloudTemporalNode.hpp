#pragma once

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

	struct CloudTemporalPushConstants {
		std::uint32_t colorIndex{0};
		std::uint32_t transmittanceIndex{0};
		std::uint32_t depthIndex{0};
		std::uint32_t velocityIndex{0};
		std::uint32_t boundingMapIndex{0}; // F2: CloudBoundingMap, real shell-hit/has-density gating
		std::uint32_t historyColorIndex{0};
		std::uint32_t historyTransmittanceIndex{0};
		std::uint32_t historyDepthIndex{0}; // F2: History<CloudAccumulatedDepth>
		std::uint32_t historyMomentsIndex{0};
		std::uint32_t outColorStorageIdx{0};
		std::uint32_t outTransmittanceStorageIdx{0};
		std::uint32_t outDepthStorageIdx{0};
		std::uint32_t outMomentsStorageIdx{0};
		std::uint32_t errorMapBufferIdx{0}; // Stage D: tile-freshness check, see cloud_temporal.comp
		std::uint32_t tileCols{0};
		std::uint32_t outputWidth{0};
		std::uint32_t outputHeight{0};
	};

	// F2: real rewrite matching boidish's cloud_temporal_reprojection.comp structurally --
	// reprojects+accumulates CloudRenderNode's fresh per-frame raymarch (CloudColor/
	// CloudTransmittance/CloudDepth/CloudVelocity) against its own prior output
	// (ReadPrevious<CloudAccumulatedColor/Transmittance/Depth/Moments>, the self-referential
	// ping-pong ResourceKeys.hpp's comment describes) into this frame's accumulated buffers --
	// see shaders/cloud_temporal.comp for the real multi-signal rejection/YCoCg-clip math. Also
	// now the sole writer of CloudTileErrorMap's error/flow channels (a per-8x8-tile shared-
	// memory reduction, every tile every frame, matching boidish exactly) -- Modify<>, not
	// Read<>, since CloudTileSchedulerNode/CloudRenderNode also touch this same buffer (ordering
	// already established via the other resources' accessor edges, same reasoning as elsewhere
	// in this pipeline). CloudCompositeNode reads this node's output now, not CloudRenderNode's
	// raw output directly.
	struct CloudTemporalNode: render::NodeRegistrar<CloudTemporalNode> {
		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                temporalShader; // shaders/cloud_temporal.comp
		CloudTemporalPushConstants   push{};
		std::uint32_t                width{0};
		std::uint32_t                height{0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!temporalShader.CompileComputeFromFile(services.device, "shaders/cloud_temporal.comp")) {
				spdlog::critical("CloudTemporalNode shader compilation failed.");
				throw std::runtime_error("CloudTemporalNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&temporalShader);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<CloudAccumulatedColor>();
				catalog.Get<CloudAccumulatedTransmittance>();
				catalog.Get<CloudAccumulatedDepth>();
				catalog.Get<CloudMoments>();
			}
		}

		void Destroy(vk::Device device) { temporalShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<CloudColor>();
			r.Read<CloudTransmittance>();
			r.Read<CloudDepth>();
			r.Read<CloudVelocity>();
			r.Read<CloudBoundingMap>();
			r.Create<CloudAccumulatedColor>();
			r.ReadPrevious<CloudAccumulatedColor>();
			r.Create<CloudAccumulatedTransmittance>();
			r.ReadPrevious<CloudAccumulatedTransmittance>();
			r.Create<CloudAccumulatedDepth>();
			r.ReadPrevious<CloudAccumulatedDepth>();
			r.Create<CloudMoments>();
			r.ReadPrevious<CloudMoments>();
			r.Modify<CloudTileErrorMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.colorIndex = ctx.Index<CloudColor>();
			push.transmittanceIndex = ctx.Index<CloudTransmittance>();
			push.depthIndex = ctx.Index<CloudDepth>();
			push.velocityIndex = ctx.Index<CloudVelocity>();
			push.boundingMapIndex = ctx.Index<CloudBoundingMap>();
			push.historyColorIndex = ctx.Index<graph::History<CloudAccumulatedColor>>();
			push.historyTransmittanceIndex = ctx.Index<graph::History<CloudAccumulatedTransmittance>>();
			push.historyDepthIndex = ctx.Index<graph::History<CloudAccumulatedDepth>>();
			push.historyMomentsIndex = ctx.Index<graph::History<CloudMoments>>();
			push.outColorStorageIdx = ctx.StorageIndex<CloudAccumulatedColor>();
			push.outTransmittanceStorageIdx = ctx.StorageIndex<CloudAccumulatedTransmittance>();
			push.outDepthStorageIdx = ctx.StorageIndex<CloudAccumulatedDepth>();
			push.outMomentsStorageIdx = ctx.StorageIndex<CloudMoments>();
			push.errorMapBufferIdx = ctx.BufferIndex<CloudTileErrorMap>();
			push.tileCols = (width + 7) / 8;
			push.outputWidth = width;
			push.outputHeight = height;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, temporalShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((width + 7) / 8, (height + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudTemporalNode);

} // namespace brassica
