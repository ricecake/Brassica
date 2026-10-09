#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

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
#include "spdlog/spdlog.h"

namespace brassica {

	namespace constants::Weather {
		constexpr std::uint32_t kSimTextureDim = 1024;
		constexpr std::uint32_t kCoverageTextureDim = 2048;
		constexpr std::uint32_t kBiomeTextureDim = 2048;
		constexpr std::uint32_t kBlendDitherDivisor = 8;
		constexpr std::uint32_t kUpdateInterval = 300;
	}

	struct ClimateInitialPushConstants {
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kSimTextureDim};
	};

	struct ClimateAdvectPushConstants {
		std::uint32_t inStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kSimTextureDim};
		float         dt{1.0f};
	};

	struct ClimateBFECCCorrectPushConstants {
		std::uint32_t forwardStorageIdx{0};
		std::uint32_t originalStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kSimTextureDim};
		float         dt{1.0f};
	};

	struct ClimateWeatherPushConstants {
		std::uint32_t climateStorageIdx{0};
		std::uint32_t weatherOutStorageIdx{0};
		std::uint32_t biomeOutStorageIdx{0};
		std::uint32_t simTextureDim{constants::Weather::kSimTextureDim};
		std::uint32_t outTextureDim{constants::Weather::kCoverageTextureDim};
		std::uint32_t pad{0};
	};

	struct ClimateBlendPushConstants {
		std::uint32_t inWeatherMapAStorageIdx{0};
		std::uint32_t inWeatherMapBStorageIdx{0};
		std::uint32_t outWeatherStorageIdx{0};
		std::uint32_t inBiomeMapAStorageIdx{0};
		std::uint32_t inBiomeMapBStorageIdx{0};
		std::uint32_t outBiomeStorageIdx{0};
		std::uint32_t textureDim{constants::Weather::kCoverageTextureDim};
		float         blendFactor{0.0f};
		std::uint32_t frameCounter{0};
		std::uint32_t ditherDivisor{constants::Weather::kBlendDitherDivisor};
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
			graph::Create<TerrainBiomeTexture>,
			graph::Create<TerrainBiomeMapATexture>,
			graph::Create<TerrainBiomeMapBTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            initialShader;
		ComputeShader            advectShader;
		ComputeShader            correctShader;
		ComputeShader            weatherShader;
		ComputeShader            blendShader;

		bool          forceRegeneration{true};
		bool          hasEverGenerated{false};
		std::uint32_t frameCounter{0};
		std::uint32_t lastSimFrame{0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!initialShader.CompileComputeFromFile(services.device, "shaders/climate_initial.comp") ||
			    !advectShader.CompileComputeFromFile(services.device, "shaders/climate_advect.comp") ||
			    !correctShader.CompileComputeFromFile(services.device, "shaders/climate_bfecc_correct.comp") ||
			    !weatherShader.CompileComputeFromFile(services.device, "shaders/climate_weather.comp") ||
			    !blendShader.CompileComputeFromFile(services.device, "shaders/climate_blend.comp")) {
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
			watcher.RegisterShader(&blendShader);
		}

		void Destroy(vk::Device device) {
			initialShader.Destroy(device);
			advectShader.Destroy(device);
			correctShader.Destroy(device);
			weatherShader.Destroy(device);
			blendShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			forceRegeneration = p.forceRegeneration;
			++frameCounter;
		}

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = true
			};
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
					.key = graph::IdOf<TerrainBiomeTexture>(),
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
			vk::CommandBuffer                              vkCmd,
			ComputeShader&                                 shader,
			const std::array<vk::DescriptorSetLayout, 2>&  setLayouts,
			const std::array<vk::DescriptorSet, 2>&        boundSets,
			const PushT&                                   push,
			std::uint32_t                                  dispatchDim
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
			std::uint32_t pingPongSimIdx = ctx.StorageIndex<TerrainWeatherPingPongTexture>();
			std::uint32_t weatherMapAIdx = ctx.StorageIndex<TerrainWeatherMapATexture>();
			std::uint32_t weatherMapBIdx = ctx.StorageIndex<TerrainWeatherMapBTexture>();
			std::uint32_t mainBiomeIdx   = ctx.StorageIndex<TerrainBiomeTexture>();
			std::uint32_t biomeMapAIdx   = ctx.StorageIndex<TerrainBiomeMapATexture>();
			std::uint32_t biomeMapBIdx   = ctx.StorageIndex<TerrainBiomeMapBTexture>();

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

			std::uint32_t cycleFrame = (frameCounter % constants::Weather::kUpdateInterval);

			if (!hasEverGenerated || forceRegeneration) {
				// Initial baseline generation
				DispatchCompute(
					vkCmd,
					initialShader,
					setLayouts,
					boundSets,
					ClimateInitialPushConstants{.outStorageIdx = pingPongSimIdx, .textureDim = constants::Weather::kSimTextureDim},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				// Advect simulation state
				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = pingPongSimIdx, .outStorageIdx = mainWeatherIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 1.0f
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					correctShader,
					setLayouts,
					boundSets,
					ClimateBFECCCorrectPushConstants{
						.forwardStorageIdx = mainWeatherIdx,
						.originalStorageIdx = pingPongSimIdx,
						.outStorageIdx = pingPongSimIdx,
						.textureDim = constants::Weather::kSimTextureDim,
						.dt = 1.0f,
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = pingPongSimIdx, .outStorageIdx = mainWeatherIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 1.0f
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = mainWeatherIdx, .outStorageIdx = pingPongSimIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 0.0f
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				// Output initial weather and biome maps to Map A and Map B
				DispatchCompute(
					vkCmd,
					weatherShader,
					setLayouts,
					boundSets,
					ClimateWeatherPushConstants{
						.climateStorageIdx = pingPongSimIdx,
						.weatherOutStorageIdx = weatherMapBIdx,
						.biomeOutStorageIdx = biomeMapBIdx,
						.simTextureDim = constants::Weather::kSimTextureDim,
						.outTextureDim = constants::Weather::kCoverageTextureDim
					},
					constants::Weather::kCoverageTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					weatherShader,
					setLayouts,
					boundSets,
					ClimateWeatherPushConstants{
						.climateStorageIdx = pingPongSimIdx,
						.weatherOutStorageIdx = weatherMapAIdx,
						.biomeOutStorageIdx = biomeMapAIdx,
						.simTextureDim = constants::Weather::kSimTextureDim,
						.outTextureDim = constants::Weather::kCoverageTextureDim
					},
					constants::Weather::kCoverageTextureDim
				);
				insertComputeBarrier(vkCmd);

				hasEverGenerated = true;
				lastSimFrame = frameCounter;
			} else if (cycleFrame == 1) {
				// Staggered step 1: Before simulation, promote Map B target to Map A start
				DispatchCompute(
					vkCmd,
					blendShader,
					setLayouts,
					boundSets,
					ClimateBlendPushConstants{
						.inWeatherMapAStorageIdx = weatherMapAIdx,
						.inWeatherMapBStorageIdx = weatherMapBIdx,
						.outWeatherStorageIdx = weatherMapAIdx,
						.inBiomeMapAStorageIdx = biomeMapAIdx,
						.inBiomeMapBStorageIdx = biomeMapBIdx,
						.outBiomeStorageIdx = biomeMapAIdx,
						.textureDim = constants::Weather::kCoverageTextureDim,
						.blendFactor = 1.0f,
						.frameCounter = 0,
						.ditherDivisor = 1
					},
					constants::Weather::kCoverageTextureDim
				);
				insertComputeBarrier(vkCmd);

				// Execute simulation advection on 1024x1024 simulation state
				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = pingPongSimIdx, .outStorageIdx = mainWeatherIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 1.0f
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					correctShader,
					setLayouts,
					boundSets,
					ClimateBFECCCorrectPushConstants{
						.forwardStorageIdx = mainWeatherIdx,
						.originalStorageIdx = pingPongSimIdx,
						.outStorageIdx = pingPongSimIdx,
						.textureDim = constants::Weather::kSimTextureDim,
						.dt = 1.0f,
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = pingPongSimIdx, .outStorageIdx = mainWeatherIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 1.0f
					},
					constants::Weather::kSimTextureDim
				);
				insertComputeBarrier(vkCmd);

				DispatchCompute(
					vkCmd,
					advectShader,
					setLayouts,
					boundSets,
					ClimateAdvectPushConstants{
						.inStorageIdx = mainWeatherIdx, .outStorageIdx = pingPongSimIdx, .textureDim = constants::Weather::kSimTextureDim, .dt = 0.0f
					},
					constants::Weather::kSimTextureDim
				);
			} else if (cycleFrame == 2) {
				// Staggered step 2: Compute new weather and biome outputs into Target Map B
				DispatchCompute(
					vkCmd,
					weatherShader,
					setLayouts,
					boundSets,
					ClimateWeatherPushConstants{
						.climateStorageIdx = pingPongSimIdx,
						.weatherOutStorageIdx = weatherMapBIdx,
						.biomeOutStorageIdx = biomeMapBIdx,
						.simTextureDim = constants::Weather::kSimTextureDim,
						.outTextureDim = constants::Weather::kCoverageTextureDim
					},
					constants::Weather::kCoverageTextureDim
				);
				insertComputeBarrier(vkCmd);
				lastSimFrame = frameCounter;
			}

			// Every frame: 1/8th dithered blend pass of Map A -> Map B into main display textures
			float blendFactor = std::clamp(
				static_cast<float>(frameCounter - lastSimFrame) / static_cast<float>(constants::Weather::kUpdateInterval),
				0.0f,
				1.0f
			);

			DispatchCompute(
				vkCmd,
				blendShader,
				setLayouts,
				boundSets,
				ClimateBlendPushConstants{
					.inWeatherMapAStorageIdx = weatherMapAIdx,
					.inWeatherMapBStorageIdx = weatherMapBIdx,
					.outWeatherStorageIdx = mainWeatherIdx,
					.inBiomeMapAStorageIdx = biomeMapAIdx,
					.inBiomeMapBStorageIdx = biomeMapBIdx,
					.outBiomeStorageIdx = mainBiomeIdx,
					.textureDim = constants::Weather::kCoverageTextureDim,
					.blendFactor = blendFactor,
					.frameCounter = frameCounter,
					.ditherDivisor = constants::Weather::kBlendDitherDivisor
				},
				constants::Weather::kCoverageTextureDim
			);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
