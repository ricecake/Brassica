#pragma once

#include <glm/glm.hpp>
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
		std::uint32_t hdrColorIndex{0};
		std::uint32_t frameCounter{0};
		glm::uvec2    screenSize{0, 0};
	};

	// Derives this frame's variable-rate-shading tile map from THIS frame's own fully-composited
	// HdrColor, via real per-pixel screen-space derivatives (GL_KHR_compute_shader_derivatives --
	// see shading_rate.comp's own comment) measuring the actual visible color gradient a tile
	// contains, not a depth-based proxy: a depth discontinuity only matters if it's visually
	// perceptible, and plenty of visually perceptible detail (texture, specular, curvature
	// shading) isn't depth-driven at all. Written this frame, consumed next frame via
	// ReadPrevious<ShadingRateMap>() (same lag TerrainNode/EntityNode/ModelRenderNode/
	// DeferredNode's own tagged reads already have) -- see shading_rate.comp's own comment for
	// the dithered incremental-coarsen schedule that keeps a tile from ever getting stuck coarse.
	// Always active (no Active() gate): those four nodes all declare ReadPrevious<ShadingRateMap>()
	// unconditionally once wired, so this needs to run every frame the same way GpuLightsResetNode
	// does for GpuLightsBuffer.
	struct ShadingRateComputeNode: render::NodeRegistrar<ShadingRateComputeNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            width{0};
		std::uint32_t            height{0};

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

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			r.Domain(graph::ExecutionDomain::Compute);
			r.ReadPrevious<ShadingRateMap>();
			// Plain Read, not ReadPrevious: this frame's own fully-composited frame (the
			// compiler's "reader sees final state" default puts this after every HdrColor
			// writer this frame, clouds/water/particles included) -- see the class comment for
			// why that's the signal wanted, not a depth proxy.
			r.Read<HdrColor>();
			r.Create<ShadingRateMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (width == 0 || height == 0) {
				return;
			}

			ShadingRateComputePushConstants push{
				.prevShadingRateIndex = ctx.StorageIndex<graph::History<ShadingRateMap>>(),
				.outShadingRateIndex = ctx.StorageIndex<ShadingRateMap>(),
				.hdrColorIndex = ctx.Index<HdrColor>(),
				.frameCounter = static_cast<std::uint32_t>(ctx.frameIndex),
				.screenSize = glm::uvec2(width, height),
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			// One workgroup per tile now (local_size_x/y == TileWidth/TileHeight in
			// shading_rate.comp, one invocation per pixel) -- no rounding-up-by-8 grouping
			// needed, dispatch size is exactly the tile grid.
			const std::uint32_t tilesX = (width + s_shadingRateTileWidth - 1) / s_shadingRateTileWidth;
			const std::uint32_t tilesY = (height + s_shadingRateTileHeight - 1) / s_shadingRateTileHeight;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(tilesX, tilesY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ShadingRateComputeNode);

} // namespace brassica
