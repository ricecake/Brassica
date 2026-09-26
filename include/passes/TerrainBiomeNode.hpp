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
		bool          forceRegeneration{true};
		std::uint32_t pad{0};
	};

	struct ClimateAdvectPushConstants {
		std::uint32_t inStorageIdx{0};
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
		};
	}

	struct TerrainBiomeNode: render::NodeRegistrar<TerrainBiomeNode> {
		using Resources = graph::Declares<
			graph::Create<TerrainWeatherBiomeTexture>,
			graph::Create<TerrainWeatherPingPongTexture>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            initialShader;
		ComputeShader            advectShader;
		ComputeShader            weatherShader;

		bool          forceRegeneration{true};
		bool          hasUpdate{true};
		std::uint32_t textureDim{2048};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!initialShader.CompileComputeFromFile(services.device, "shaders/climate_initial.comp") ||
			    !advectShader.CompileComputeFromFile(services.device, "shaders/climate_advect.comp") ||
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
			watcher.RegisterShader(&weatherShader);
		}

		void Destroy(vk::Device device) {
			initialShader.Destroy(device);
			advectShader.Destroy(device);
			weatherShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			forceRegeneration = p.forceRegeneration;
			hasUpdate = (p.cameraPosition != p.previousCameraPosition);
		}

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = forceRegeneration || hasUpdate
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

			// Pass 1: Generate initial climate texture into pingPongIdx
			{
				ClimateInitialPushConstants pushInitial{
					.outStorageIdx = pingPongIdx,
					.textureDim = textureDim,
					.forceRegeneration = forceRegeneration,
				};

				std::array<vk::PushConstantRange, 1> pushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClimateInitialPushConstants)}
				};

				render::ComputePipelineRequest req{
					.shader = &initialShader,
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
				vkCmd.pushConstants(
					res.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(ClimateInitialPushConstants),
					&pushInitial
				);

				std::uint32_t groupCount = (textureDim + 15) / 16;
				vkCmd.dispatch(groupCount, groupCount, 1);
			}

			insertComputeBarrier(vkCmd);

			// Pass 2: Advection Ping-Pong Passes (4 iterations)
			std::uint32_t currentIn = pingPongIdx;
			std::uint32_t currentOut = mainIdx;

			{
				std::array<vk::PushConstantRange, 1> pushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClimateAdvectPushConstants)}
				};

				render::ComputePipelineRequest req{
					.shader = &advectShader,
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

				std::uint32_t groupCount = (textureDim + 15) / 16;

				for (int i = 0; i < 4; ++i) {
					ClimateAdvectPushConstants pushAdvect{
						.inStorageIdx = currentIn,
						.outStorageIdx = currentOut,
						.textureDim = textureDim,
						.dt = 1.0f,
					};

					vkCmd.pushConstants(
						res.layout,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sizeof(ClimateAdvectPushConstants),
						&pushAdvect
					);

					vkCmd.dispatch(groupCount, groupCount, 1);

					insertComputeBarrier(vkCmd);

					std::swap(currentIn, currentOut);
				}
			}

			// Pass 3: Weather Analysis & Whittaker Biome Pass (reads currentIn, writes mainIdx)
			{
				ClimateWeatherPushConstants pushWeather{
					.climateStorageIdx = currentIn,
					.outStorageIdx = mainIdx,
					.textureDim = textureDim,
				};

				std::array<vk::PushConstantRange, 1> pushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ClimateWeatherPushConstants)}
				};

				render::ComputePipelineRequest req{
					.shader = &weatherShader,
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

				vkCmd.pushConstants(
					res.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(ClimateWeatherPushConstants),
					&pushWeather
				);

				std::uint32_t groupCount = (textureDim + 15) / 16;
				vkCmd.dispatch(groupCount, groupCount, 1);
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainBiomeNode);

} // namespace brassica
