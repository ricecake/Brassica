#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	namespace constants::Weather {
		constexpr std::uint32_t kSimTextureDim = 2048;
		constexpr std::uint32_t kCoverageTextureDim = 2048;
		constexpr std::uint32_t kBiomeTextureDim = 2048;
		constexpr std::uint32_t kBlendDitherDivisor = 8;
		constexpr std::uint32_t kUpdateInterval = 300;
	} // namespace constants::Weather

	struct ClimateInitialPushConstants {
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kSimTextureDim};
	};

	struct ClimateCAPushConstants {
		std::uint32_t inMapStorageIdx{0};
		std::uint32_t outMapStorageIdx{0};
		std::uint32_t weatherOutStorageIdx{0};
		std::uint32_t biomeOutStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kCoverageTextureDim};
		std::uint32_t pad{0};
	};

	inline graph::ResourceDesc WeatherBiomeImageDesc(std::uint32_t dim) {
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
			graph::Create<TerrainWeatherPingPongTexture>,
			graph::Create<TerrainWeatherMapATexture>,
			graph::Create<TerrainWeatherMapBTexture>,
			graph::Create<TerrainBiomeMainTexture>,
			graph::Create<TerrainBiomeMapATexture>,
			graph::Create<TerrainBiomeMapBTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

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

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = true};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherBiomeTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherPingPongTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kSimTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherMapATexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherMapBTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kCoverageTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainBiomeMainTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainBiomeMapATexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainBiomeMapBTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(constants::Weather::kBiomeTextureDim),
				}
			);
			return r;
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
						.outStorageIdx = weatherMapAIdx,
						.textureDim = constants::Weather::kCoverageTextureDim
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
						.inMapStorageIdx = weatherMapAIdx,
						.outMapStorageIdx = weatherMapBIdx,
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
				std::uint32_t inStateIdx = pingPongState ? weatherMapBIdx : weatherMapAIdx;
				std::uint32_t outStateIdx = pingPongState ? weatherMapAIdx : weatherMapBIdx;

				// Run cellular automata step
				DispatchCompute(
					vkCmd,
					caShader,
					setLayouts,
					boundSets,
					ClimateCAPushConstants{
						.inMapStorageIdx = inStateIdx,
						.outMapStorageIdx = outStateIdx,
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
