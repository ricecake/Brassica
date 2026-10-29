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

	struct CloudTileSchedulerPushConstants {
		std::uint32_t tileQueueBufferIdx{0};
		std::uint32_t dispatchArgsBufferIdx{0};
		std::uint32_t errorMapBufferIdx{0};
		std::uint32_t boundingMapIndex{0}; // sampled: CloudBoundingMap, this frame's (not History<>)
		std::uint32_t tileCols{0};
		std::uint32_t tileRows{0};
		std::uint32_t maxTilesPerFrame{0};
		std::uint32_t spatialUpdateFrames{16}; // every tile gets a real raymarch at least this often
		std::uint32_t phase{0};
	};

	// F3: real rewrite matching boidish's cloud_tile_scheduler.comp exactly -- builds the
	// per-frame tile queue CloudRenderNode's raymarch indirect-dispatches over. See
	// shaders/cloud_tile_scheduler.comp for the real 2-pass priority scheduler: Owen-scrambled-
	// Morton periodic refresh (phase 1, no error-map touch) + real bounding-map-driven
	// densityDeltaScore/newlyAppeared + error/flow/age priority scoring (phase 2) against a
	// temporally-dithered budget threshold, plus an adaptive per-tile step multiplier
	// CloudRenderNode's raymarch reads back. 3 in-node hand-rolled-barrier dispatches (reset ->
	// spatial -> priority, same multi-dispatch-one-node pattern as CloudSpatialFilterNode's
	// 2-pass a-trous). CloudTileErrorMap persists across frames without History<> pairing -- no
	// Create<> call for it, same as ParticleBuffer's own cross-frame Modify<>-only convention
	// (see ParticleLivenessNode.hpp's comment on why that's sufficient).
	struct CloudTileSchedulerNode: render::NodeRegistrar<CloudTileSchedulerNode> {
		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   schedulerShader; // shaders/cloud_tile_scheduler.comp
		CloudTileSchedulerPushConstants push{};
		std::uint32_t                   tileCols{0};
		std::uint32_t                   tileRows{0};
		// boidish's own cloud_max_refresh_rate_ default (AtmosphereEffect.h) -- the fraction of
		// tiles the frame budget allows per frame.
		float maxTileBudgetFraction{0.25f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!schedulerShader.CompileComputeFromFile(services.device, "shaders/cloud_tile_scheduler.comp")) {
				spdlog::critical("CloudTileSchedulerNode shader compilation failed.");
				throw std::runtime_error("CloudTileSchedulerNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&schedulerShader);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<CloudTileQueue>();
				catalog.Get<CloudTileDispatchArgs>();
				catalog.Get<CloudTileErrorMap>();
			}
		}

		void Destroy(vk::Device device) { schedulerShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			tileCols = CloudTileCols(ctx);
			tileRows = CloudTileRows(ctx);

			r.Domain(graph::ExecutionDomain::Compute);
			// This frame's bounding map, not last frame's -- CloudBoundingNode runs earlier in
			// this same frame (ordered via this same Read<>() accessor edge against its sole
			// Modify<>()), so a plain Read<>() is correct here, not ReadPrevious<>().
			r.Read<CloudBoundingMap>();
			r.Create<CloudTileQueue>();
			r.Create<CloudTileDispatchArgs>();
			r.Modify<CloudTileErrorMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::uint32_t totalTiles = tileCols * tileRows;
			push.tileQueueBufferIdx = ctx.BufferIndex<CloudTileQueue>();
			push.dispatchArgsBufferIdx = ctx.BufferIndex<CloudTileDispatchArgs>();
			push.errorMapBufferIdx = ctx.BufferIndex<CloudTileErrorMap>();
			push.boundingMapIndex = ctx.Index<CloudBoundingMap>();
			push.tileCols = tileCols;
			push.tileRows = tileRows;
			push.maxTilesPerFrame = (std::max)(
				1u, static_cast<std::uint32_t>(static_cast<float>(totalTiles) * maxTileBudgetFraction)
			);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t      groupsX = (tileCols + 7) / 8;
			std::uint32_t      groupsY = (tileRows + 7) / 8;

			vk::MemoryBarrier2 barrier(
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite
			);
			vk::DependencyInfo dep({}, 1, &barrier);

			push.phase = 0; // PHASE_RESET
			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, schedulerShader, push);
			if (!resolved.pipeline) {
				return;
			}
			vkCmd.dispatch(1, 1, 1);
			vkCmd.pipelineBarrier2(dep);

			push.phase = 1; // PHASE_SPATIAL
			resolved = render::BindCompute(ctx, *pipelineLibrary, schedulerShader, push);
			if (!resolved.pipeline) {
				return;
			}
			vkCmd.dispatch(groupsX, groupsY, 1);
			vkCmd.pipelineBarrier2(dep);

			push.phase = 2; // PHASE_PRIORITY
			resolved = render::BindCompute(ctx, *pipelineLibrary, schedulerShader, push);
			if (!resolved.pipeline) {
				return;
			}
			vkCmd.dispatch(groupsX, groupsY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudTileSchedulerNode);

} // namespace brassica
