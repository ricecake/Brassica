#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	struct ClimateInitialPushConstants {
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{2048};
	};

	struct ClimateAdvectPushConstants {
		std::uint32_t inStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{2048};
		float         dt{1.0f};
	};

	struct ClimateBFECCCorrectPushConstants {
		std::uint32_t forwardStorageIdx{0};
		std::uint32_t originalStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{2048};
		float         dt{1.0f};
	};

	struct ClimateWeatherPushConstants {
		std::uint32_t climateStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{2048};
		std::uint32_t pad{0};
	};

	// Marked persistent: this field is read every time a node Creates/Modifies either weather
	// texture, so both get the lazy-allocate-once, alias-pool-bypassing treatment uniformly --
	// see ResourceDesc::persistent (Execution.hpp). Without it, the registry's alias pool would
	// eventually believe the memory backing one of these is free (it only ever runs once every
	// updateInterval frames) and hand it to an unrelated transient resource.
	inline graph::ResourceDesc WeatherBiomeImageDesc(std::uint32_t dim = 2048) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = dim,
			.height = dim,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
			),
			.persistent = true,
		};
	}

	struct TerrainBiomeNode: render::NodeRegistrar<TerrainBiomeNode> {
		using Resources = graph::Declares<
			graph::Create<TerrainWeatherBiomeTexture>,
			graph::Create<TerrainWeatherPingPongTexture>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            initialShader;
		ComputeShader            advectShader;
		ComputeShader            correctShader;
		ComputeShader            weatherShader;

		bool          forceRegeneration{true};
		bool          hasEverGenerated{false};
		std::uint32_t frameCounter{0};
		std::uint32_t updateInterval{300};
		std::uint32_t textureDim{2048};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!initialShader.CompileComputeFromFile(services.device, "shaders/climate_initial.comp") ||
			    !advectShader.CompileComputeFromFile(services.device, "shaders/climate_advect.comp") ||
			    !correctShader.CompileComputeFromFile(services.device, "shaders/climate_bfecc_correct.comp") ||
			    !weatherShader.CompileComputeFromFile(services.device, "shaders/climate_weather.comp")) {
				spdlog::critical("TerrainBiomeNode shader compilation failed.");
				throw std::runtime_error("TerrainBiomeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&initialShader);
			watcher.RegisterShader(&advectShader);
			watcher.RegisterShader(&correctShader);
			watcher.RegisterShader(&weatherShader);
		}

		void Destroy(vk::Device device) {
			initialShader.Destroy(device);
			advectShader.Destroy(device);
			correctShader.Destroy(device);
			weatherShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			forceRegeneration = p.forceRegeneration;
			++frameCounter;
		}

		graph::Recipe Setup(const graph::FrameContext&) {
			bool shouldRun = forceRegeneration || (frameCounter == 1) || (frameCounter % updateInterval == 0);
			graph::Recipe r{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = shouldRun
			};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherBiomeTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(textureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherPingPongTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(textureDim),
				}
			);
			return r;
		}

		// Binds/pushes/dispatches one of this node's compute passes. Shared across every pass
		// below -- the bind-pipeline/bind-sets/push-constants/dispatch shape is otherwise
		// identical for all five, differing only in which shader and push-constant type.
		template <typename PushT>
		void DispatchCompute(
			vk::CommandBuffer                              vkCmd,
			ComputeShader&                                 shader,
			const std::array<vk::DescriptorSetLayout, 2>&  setLayouts,
			const std::array<vk::DescriptorSet, 2>&        boundSets,
			const PushT&                                   push
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

			std::uint32_t groupCount = (textureDim + 15) / 16;
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

			// mainIdx (TerrainWeatherBiomeTexture) and pingPongIdx (TerrainWeatherPingPongTexture)
			// are both marked persistent (WeatherBiomeImageDesc) -- real GPU memory that survives
			// every frame this node is inactive, not just the frame it was allocated on. pingPongIdx
			// holds the persistent *raw* climate state (temp, moisture, windX, windY) carried
			// across activations; mainIdx ends this function holding the display weather/biome
			// texture CirrusNode/DeferredNode sample.
			std::uint32_t mainIdx = ctx.StorageIndex<TerrainWeatherBiomeTexture>();
			std::uint32_t pingPongIdx = ctx.StorageIndex<TerrainWeatherPingPongTexture>();

			auto insertComputeBarrier = [&](vk::CommandBuffer cmd) {
				vk::MemoryBarrier2 barrier{};
				barrier.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader)
					.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite)
					.setDstStageMask(vk::PipelineStageFlagBits2::eComputeShader)
					.setDstAccessMask(vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite);

				vk::DependencyInfo depInfo{};
				depInfo.setMemoryBarriers(barrier);
				cmd.pipelineBarrier2(depInfo);
			};

			// Regenerate the baseline only on first-ever generation or an explicit
			// forceRegeneration. Every other activation continues advecting whatever state
			// survived in pingPongIdx from the previous one -- that's the entire point of marking
			// these textures persistent instead of letting this unconditionally reset the
			// simulation back to the deterministic baseline field every updateInterval frames.
			if (!hasEverGenerated || forceRegeneration) {
				DispatchCompute(
					vkCmd,
					initialShader,
					setLayouts,
					boundSets,
					ClimateInitialPushConstants{.outStorageIdx = pingPongIdx, .textureDim = textureDim}
				);
				insertComputeBarrier(vkCmd);
				hasEverGenerated = true;
			}

			// Back and Forth Error Compensation and Correction (BFECC), using the two persistent
			// textures as the only two physical buffers for the whole cycle:
			//   1. forward semi-Lagrangian step:        pingPongIdx (phi0) -> mainIdx (phi1)
			//   2. backward step + pointwise correction, written in place over pingPongIdx --
			//      phi0's own texel is never backtraced there, so overwriting it in place is safe
			//      (see climate_bfecc_correct.comp)
			//   3. final forward step of the corrected field: pingPongIdx -> mainIdx (phi_final)
			//   4. relocate phi_final back to pingPongIdx (dt = 0, a pure copy -- see
			//      climate_advect.comp) so it's there for the next activation's step 1
			//   5. weather/Whittaker biome derivation: pingPongIdx (phi_final) -> mainIdx (display)
			DispatchCompute(
				vkCmd,
				advectShader,
				setLayouts,
				boundSets,
				ClimateAdvectPushConstants{
					.inStorageIdx = pingPongIdx, .outStorageIdx = mainIdx, .textureDim = textureDim, .dt = 1.0f
				}
			);
			insertComputeBarrier(vkCmd);

			DispatchCompute(
				vkCmd,
				correctShader,
				setLayouts,
				boundSets,
				ClimateBFECCCorrectPushConstants{
					.forwardStorageIdx = mainIdx,
					.originalStorageIdx = pingPongIdx,
					.outStorageIdx = pingPongIdx,
					.textureDim = textureDim,
					.dt = 1.0f,
				}
			);
			insertComputeBarrier(vkCmd);

			DispatchCompute(
				vkCmd,
				advectShader,
				setLayouts,
				boundSets,
				ClimateAdvectPushConstants{
					.inStorageIdx = pingPongIdx, .outStorageIdx = mainIdx, .textureDim = textureDim, .dt = 1.0f
				}
			);
			insertComputeBarrier(vkCmd);

			DispatchCompute(
				vkCmd,
				advectShader,
				setLayouts,
				boundSets,
				ClimateAdvectPushConstants{
					.inStorageIdx = mainIdx, .outStorageIdx = pingPongIdx, .textureDim = textureDim, .dt = 0.0f
				}
			);
			insertComputeBarrier(vkCmd);

			DispatchCompute(
				vkCmd,
				weatherShader,
				setLayouts,
				boundSets,
				ClimateWeatherPushConstants{.climateStorageIdx = pingPongIdx, .outStorageIdx = mainIdx, .textureDim = textureDim}
			);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
