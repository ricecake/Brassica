#pragma once

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/scene/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "resources/core.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct ShadingRateComputePushConstants {
		std::uint32_t prevShadingRateIndex{0};
		std::uint32_t outShadingRateIndex{0};
		std::uint32_t prevHiZIndex{0};
		std::uint32_t frameCounter{0};
		std::uint32_t tileWidth{constants::Class::ShadingRate::TileWidth};
		std::uint32_t tileHeight{constants::Class::ShadingRate::TileHeight};
	};

	// Derives this frame's variable-rate-shading tile map entirely from previous-frame data
	// (ShadingRateMap's own History pair, plus HiZTexture's History pair for a depth-discontinuity
	// detail signal) -- see shading_rate.comp's own comment for the dithered incremental-coarsen
	// schedule that keeps a tile from ever getting stuck coarse. Always active (no Active() gate):
	// DeferredNode/TerrainNode/GltfRenderNode/EntityNode all declare ReadPrevious<ShadingRateMap>()
	// unconditionally once wired, so this needs to run every frame the same way GpuLightsResetNode
	// does for GpuLightsBuffer.
	struct ShadingRateComputeNode: render::NodeRegistrar<ShadingRateComputeNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/shading_rate.comp")) {
				spdlog::critical("ShadingRateComputeNode shader compilation failed.");
				throw std::runtime_error("ShadingRateComputeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.ReadPrevious<ShadingRateMap>();
			r.ReadPrevious<HiZTexture>();
			r.Create<ShadingRateMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			ShadingRateComputePushConstants push{
				.prevShadingRateIndex = ctx.StorageIndex<graph::History<ShadingRateMap>>(),
				.outShadingRateIndex = ctx.StorageIndex<ShadingRateMap>(),
				.prevHiZIndex = ctx.Index<graph::History<HiZTexture>>(),
				.frameCounter = static_cast<std::uint32_t>(ctx.frameIndex),
				.tileWidth = s_shadingRateTileWidth,
				.tileHeight = s_shadingRateTileHeight,
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			const std::uint32_t tilesX = (ctx.width + s_shadingRateTileWidth - 1) / s_shadingRateTileWidth;
			const std::uint32_t tilesY = (ctx.height + s_shadingRateTileHeight - 1) / s_shadingRateTileHeight;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((tilesX + 7) / 8, (tilesY + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ShadingRateComputeNode);

} // namespace brassica
