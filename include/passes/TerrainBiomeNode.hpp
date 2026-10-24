#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// constants::Weather and WeatherBiomeImageDesc moved to ResourceKeys.hpp (TerrainBiomeTexture
	// etc.'s Desc() needs them, and ResourceKeys.hpp can't include this file back) -- still
	// reachable here through the existing #include above, nothing below changes. kSimTextureDim
	// is 2048 there too (not the stale 1024 this namespace had before the move) -- origin/main
	// independently bumped it to match kCoverageTextureDim/kBiomeTextureDim while this was moved;
	// see ResourceKeys.hpp.

	struct ClimateInitialPushConstants {
		std::uint32_t outStorageIdx1{0};
		std::uint32_t outStorageIdx2{0};
		std::uint32_t textureDim{constants::Weather::kCoverageTextureDim};
		std::uint32_t pad{0};
	};

	struct ClimateCAPushConstants {
		std::uint32_t inMapStorageIdx1{0};
		std::uint32_t inMapStorageIdx2{0};
		std::uint32_t outMapStorageIdx1{0};
		std::uint32_t outMapStorageIdx2{0};
		std::uint32_t weatherOutStorageIdx{0};
		std::uint32_t biomeOutStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kCoverageTextureDim};
		std::uint32_t cellOffset{0};
		std::uint32_t cellsPerUpdate{constants::Weather::kCellsPerUpdate};
		std::uint32_t pad{0};
	};

	// S4: no kPhase -- this node is the sole writer of all seven of its weather/biome keys (six
	// of them touched nowhere else in the codebase at all), so there's nothing left for a phase
	// to order it against; every reader of TerrainWeatherBiomeTexture (the one key with other
	// touchers) gets a real resource edge from this node's Modify<> instead.
	struct TerrainBiomeNode: render::NodeRegistrar<TerrainBiomeNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            initialShader;
		ComputeShader            caShader;

		bool          forceRegeneration{true};
		bool          hasEverGenerated{false};
		bool          pingPongState{false};
		std::uint32_t frameCounter{0};
		std::uint32_t cellOffset{0};
		std::uint32_t cellsPerUpdate{constants::Weather::kCellsPerUpdate};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!initialShader.CompileComputeFromFile(services.device, "shaders/climate_initial.comp") ||
			    !caShader.CompileComputeFromFile(services.device, "shaders/climate_ca.comp")) {
				spdlog::critical("TerrainBiomeNode shader compilation failed.");
				throw std::runtime_error("TerrainBiomeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			if (services.physicalRegistry) {
				graph::ResourceCatalog& catalog = services.physicalRegistry->Catalog();
				catalog.Get<TerrainWeatherBiomeTexture>();
				catalog.Get<TerrainWeatherPingPongTexture>();
				catalog.Get<TerrainWeatherMapATexture>();
				catalog.Get<TerrainWeatherMapBTexture>();
				catalog.Get<TerrainBiomeMainTexture>();
				catalog.Get<TerrainBiomeMapATexture>();
				catalog.Get<TerrainBiomeMapBTexture>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&initialShader);
			watcher.RegisterShader(&caShader);
		}

		void Destroy(vk::Device device) {
			initialShader.Destroy(device);
			caShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			forceRegeneration = p.forceRegeneration;
			++frameCounter;
		}

		// All seven are Modify (not Create): legacy used ReadWrite for every one of these, since
		// each is persistent ping-pong/display state read back in by this same node next frame,
		// not a fresh-every-frame allocation. Dims now come from each key's own catalog Desc()
		// (ResourceKeys.hpp's WeatherBiomeImageDesc) instead of this Setup rebuilding them by hand.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Modify<TerrainWeatherBiomeTexture>();
			r.Modify<TerrainWeatherPingPongTexture>();
			r.Modify<TerrainWeatherMapATexture>();
			r.Modify<TerrainWeatherMapBTexture>();
			r.Modify<TerrainBiomeMainTexture>();
			r.Modify<TerrainBiomeMapATexture>();
			r.Modify<TerrainBiomeMapBTexture>();
			// shaders/climate_ca.comp reads uLights[0].direction via clustered_lighting.glsl.
			r.Read<LightsBuffer>();
		}

		// Shared square-workgroup dispatch math for all three CA passes below -- bind/push itself
		// comes from render::BindCompute, this just adds the one piece genuinely specific to this
		// node's shaders (dispatchDim -> group count).
		template <typename PushT>
		void
		DispatchCompute2D(
			graph::NodeContext& ctx,
			ComputeShader&      shader,
			const PushT&        push,
			std::uint32_t       dispatchDimX,
			std::uint32_t       dispatchDimY
		) {
			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, shader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t     groupCountX = (dispatchDimX + 15) / 16;
			std::uint32_t     groupCountY = (dispatchDimY + 15) / 16;
			vkCmd.dispatch(groupCountX, groupCountY, 1);
		}

		template <typename PushT>
		void
		DispatchCompute(graph::NodeContext& ctx, ComputeShader& shader, const PushT& push, std::uint32_t dispatchDim) {
			DispatchCompute2D(ctx, shader, push, dispatchDim, dispatchDim);
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			std::uint32_t mainWeatherIdx = ctx.StorageIndex<TerrainWeatherBiomeTexture>();
			std::uint32_t weatherMapAIdx = ctx.StorageIndex<TerrainWeatherMapATexture>();
			std::uint32_t weatherMapBIdx = ctx.StorageIndex<TerrainWeatherMapBTexture>();
			std::uint32_t biomeMapAIdx = ctx.StorageIndex<TerrainBiomeMapATexture>();
			std::uint32_t biomeMapBIdx = ctx.StorageIndex<TerrainBiomeMapBTexture>();
			std::uint32_t mainBiomeIdx = ctx.StorageIndex<TerrainBiomeMainTexture>();

			auto insertComputeBarrier = [&](vk::CommandBuffer cmd) {
				vk::MemoryBarrier2 barrier{};
				barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
					.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
					.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
					.setDstAccessMask(
						vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite
					);

				vk::DependencyInfo depInfo{};
				depInfo.setMemoryBarriers(barrier);
				cmd.pipelineBarrier2(depInfo);
			};

			const std::uint32_t textureDim = constants::Weather::kCoverageTextureDim;

			if (!hasEverGenerated || forceRegeneration) {
				// Initial baseline CA generation
				DispatchCompute(
					ctx,
					initialShader,
					ClimateInitialPushConstants{
						.outStorageIdx1 = weatherMapAIdx,
						.outStorageIdx2 = biomeMapAIdx,
						.textureDim = textureDim,
						.pad = 0
					},
					textureDim
				);
				insertComputeBarrier(vkCmd);

				cellOffset = 0;
				std::uint32_t cellsToProcess = std::min(cellsPerUpdate, textureDim - cellOffset);

				// First Cellular Automata partial update step from Map A -> Map B, writing outputs to main weather and biome maps
				DispatchCompute2D(
					ctx,
					caShader,
					ClimateCAPushConstants{
						.inMapStorageIdx1 = weatherMapAIdx,
						.inMapStorageIdx2 = biomeMapAIdx,
						.outMapStorageIdx1 = weatherMapBIdx,
						.outMapStorageIdx2 = biomeMapBIdx,
						.weatherOutStorageIdx = mainWeatherIdx,
						.biomeOutStorageIdx = mainBiomeIdx,
						.textureDim = textureDim,
						.cellOffset = cellOffset,
						.cellsPerUpdate = cellsToProcess,
						.pad = 0
					},
					textureDim,
					cellsToProcess
				);

				cellOffset += cellsToProcess;
				if (cellOffset >= textureDim) {
					cellOffset = 0;
					pingPongState = true;
				} else {
					pingPongState = false;
				}
				hasEverGenerated = true;
			} else {
				std::uint32_t inStateIdx1 = pingPongState ? weatherMapBIdx : weatherMapAIdx;
				std::uint32_t inStateIdx2 = pingPongState ? biomeMapBIdx : biomeMapAIdx;
				std::uint32_t outStateIdx1 = pingPongState ? weatherMapAIdx : weatherMapBIdx;
				std::uint32_t outStateIdx2 = pingPongState ? biomeMapAIdx : biomeMapBIdx;

				std::uint32_t cellsToProcess = std::min(cellsPerUpdate, textureDim - cellOffset);

				// Run cellular automata step for fixed subset of cells
				DispatchCompute2D(
					ctx,
					caShader,
					ClimateCAPushConstants{
						.inMapStorageIdx1 = inStateIdx1,
						.inMapStorageIdx2 = inStateIdx2,
						.outMapStorageIdx1 = outStateIdx1,
						.outMapStorageIdx2 = outStateIdx2,
						.weatherOutStorageIdx = mainWeatherIdx,
						.biomeOutStorageIdx = mainBiomeIdx,
						.textureDim = textureDim,
						.cellOffset = cellOffset,
						.cellsPerUpdate = cellsToProcess,
						.pad = 0
					},
					textureDim,
					cellsToProcess
				);

				cellOffset += cellsToProcess;
				if (cellOffset >= textureDim) {
					cellOffset = 0;
					pingPongState = !pingPongState;
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
