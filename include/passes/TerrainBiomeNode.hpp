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

	struct ClimateInitialPushConstants {
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{4096};
	};

	struct ClimateAdvectPushConstants {
		std::uint32_t inStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{4096};
		float         dt{1.0f};
	};

	struct ClimateBFECCCorrectPushConstants {
		std::uint32_t forwardStorageIdx{0};
		std::uint32_t originalStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{4096};
		float         dt{1.0f};
	};

	struct ClimateWeatherPushConstants {
		std::uint32_t climateStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{4096};
		std::uint32_t pad{0};
	};

	struct ClimateBlendPushConstants {
		std::uint32_t inMapAStorageIdx{0};
		std::uint32_t inMapBStorageIdx{0};
		std::uint32_t outStorageIdx{0};
		std::uint32_t textureDim{4096};
		float         blendFactor{0.0f};
	};

	// Marked persistent: weather textures survive across frames.
	inline graph::ResourceDesc WeatherBiomeImageDesc(std::uint32_t dim = 4096) {
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
			graph::Create<TerrainWeatherMapBTexture>>;

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
		std::uint32_t updateInterval{300};
		std::uint32_t textureDim{4096};

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
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherMapATexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(textureDim),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainWeatherMapBTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = WeatherBiomeImageDesc(textureDim),
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

			std::uint32_t mainIdx = ctx.StorageIndex<TerrainWeatherBiomeTexture>();
			std::uint32_t pingPongIdx = ctx.StorageIndex<TerrainWeatherPingPongTexture>();
			std::uint32_t mapAIdx = ctx.StorageIndex<TerrainWeatherMapATexture>();
			std::uint32_t mapBIdx = ctx.StorageIndex<TerrainWeatherMapBTexture>();

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

			bool shouldRunSimulation = forceRegeneration || !hasEverGenerated || (frameCounter == 1) || (frameCounter % updateInterval == 0);

			if (shouldRunSimulation) {
				if (!hasEverGenerated || forceRegeneration) {
					DispatchCompute(
						vkCmd,
						initialShader,
						setLayouts,
						boundSets,
						ClimateInitialPushConstants{.outStorageIdx = pingPongIdx, .textureDim = textureDim}
					);
					insertComputeBarrier(vkCmd);

					// BFECC advection
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

					// Output initial weather simulation state into Map A and Map B
					DispatchCompute(
						vkCmd,
						weatherShader,
						setLayouts,
						boundSets,
						ClimateWeatherPushConstants{.climateStorageIdx = pingPongIdx, .outStorageIdx = mapBIdx, .textureDim = textureDim}
					);
					insertComputeBarrier(vkCmd);

					DispatchCompute(
						vkCmd,
						weatherShader,
						setLayouts,
						boundSets,
						ClimateWeatherPushConstants{.climateStorageIdx = pingPongIdx, .outStorageIdx = mapAIdx, .textureDim = textureDim}
					);
					insertComputeBarrier(vkCmd);

					hasEverGenerated = true;
				} else {
					// Blend Map B into Map A so Map A becomes the starting point of the new fade cycle
					DispatchCompute(
						vkCmd,
						blendShader,
						setLayouts,
						boundSets,
						ClimateBlendPushConstants{
							.inMapAStorageIdx = mapAIdx,
							.inMapBStorageIdx = mapBIdx,
							.outStorageIdx = mapAIdx,
							.textureDim = textureDim,
							.blendFactor = 1.0f,
						}
					);
					insertComputeBarrier(vkCmd);

					// Advect simulation state forward
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

					// Generate new simulation output into Map B (Target)
					DispatchCompute(
						vkCmd,
						weatherShader,
						setLayouts,
						boundSets,
						ClimateWeatherPushConstants{.climateStorageIdx = pingPongIdx, .outStorageIdx = mapBIdx, .textureDim = textureDim}
					);
					insertComputeBarrier(vkCmd);
				}
				lastSimFrame = frameCounter;
			}

			// Smoothly blend Map A into Map B over updateInterval frames
			float blendFactor = std::clamp(
				static_cast<float>(frameCounter - lastSimFrame) / static_cast<float>(updateInterval),
				0.0f,
				1.0f
			);

			DispatchCompute(
				vkCmd,
				blendShader,
				setLayouts,
				boundSets,
				ClimateBlendPushConstants{
					.inMapAStorageIdx = mapAIdx,
					.inMapBStorageIdx = mapBIdx,
					.outStorageIdx = mainIdx,
					.textureDim = textureDim,
					.blendFactor = blendFactor,
				}
			);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
