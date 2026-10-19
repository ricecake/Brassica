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
		}

		template <typename PushT>
		void DispatchCompute(
			vk::CommandBuffer                             vkCmd,
			ComputeShader&                                shader,
			const std::array<vk::DescriptorSetLayout, 2>& setLayouts,
			const std::array<vk::DescriptorSet, 2>&       boundSets,
			const PushT&                                  push,
			std::uint32_t                                 dispatchDim
		) {
			std::array<vk::PushConstantRange, 1> pushRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushT)}
			};

			render::ComputePipelineRequest req{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushRanges,
			};
			render::ResolvedPipeline res = pipelineLibrary->ResolveCached(req);

			if (res.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, res.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, res.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(res.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushT), &push);

			std::uint32_t groupCount = (dispatchDim + 15) / 16;
			vkCmd.dispatch(groupCount, groupCount, 1);
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};

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

			if (!hasEverGenerated || forceRegeneration) {
				// Initial baseline CA generation
				DispatchCompute(
					vkCmd,
					initialShader,
					setLayouts,
					boundSets,
					ClimateInitialPushConstants{
						.outStorageIdx1 = weatherMapAIdx,
						.outStorageIdx2 = biomeMapAIdx,
						.textureDim = constants::Weather::kCoverageTextureDim,
						.pad = 0
					},
					constants::Weather::kCoverageTextureDim
				);
				insertComputeBarrier(vkCmd);

				// First Cellular Automata update step from Map A -> Map B, writing outputs to main weather and biome maps
				DispatchCompute(
					vkCmd,
					caShader,
					setLayouts,
					boundSets,
					ClimateCAPushConstants{
						.inMapStorageIdx1 = weatherMapAIdx,
						.inMapStorageIdx2 = biomeMapAIdx,
						.outMapStorageIdx1 = weatherMapBIdx,
						.outMapStorageIdx2 = biomeMapBIdx,
						.weatherOutStorageIdx = mainWeatherIdx,
						.biomeOutStorageIdx = mainBiomeIdx,
						.textureDim = constants::Weather::kCoverageTextureDim,
						.pad = 0
					},
					constants::Weather::kCoverageTextureDim
				);

				pingPongState = true; // Next input will be Map B
				hasEverGenerated = true;
			} else {
				std::uint32_t inStateIdx1 = pingPongState ? weatherMapBIdx : weatherMapAIdx;
				std::uint32_t inStateIdx2 = pingPongState ? biomeMapBIdx : biomeMapAIdx;
				std::uint32_t outStateIdx1 = pingPongState ? weatherMapAIdx : weatherMapBIdx;
				std::uint32_t outStateIdx2 = pingPongState ? biomeMapAIdx : biomeMapBIdx;

				// Run cellular automata step
				DispatchCompute(
					vkCmd,
					caShader,
					setLayouts,
					boundSets,
					ClimateCAPushConstants{
						.inMapStorageIdx1 = inStateIdx1,
						.inMapStorageIdx2 = inStateIdx2,
						.outMapStorageIdx1 = outStateIdx1,
						.outMapStorageIdx2 = outStateIdx2,
						.weatherOutStorageIdx = mainWeatherIdx,
						.biomeOutStorageIdx = mainBiomeIdx,
						.textureDim = constants::Weather::kCoverageTextureDim,
						.pad = 0
					},
					constants::Weather::kCoverageTextureDim
				);

				pingPongState = !pingPongState;
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
