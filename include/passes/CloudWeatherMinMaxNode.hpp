#pragma once

#include <algorithm>
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

	struct CloudWeatherMinMaxInitPushConstants {
		std::uint32_t weatherIndex{0};
		std::uint32_t dstStorageIdx{0};
		std::uint32_t dim{0};
	};

	struct CloudWeatherMinMaxDownsamplePushConstants {
		std::uint32_t srcSampledIdx{0};
		std::uint32_t dstStorageIdx{0};
		std::uint32_t srcMip{0};
		std::uint32_t dstDim{0};
	};

	// Stage E0: builds CloudWeatherMinMaxMap's full mip chain (mip 0 = raw coverage duplicated
	// into min/max, mips 1..10 = 2x2 min/max downsample) every active frame -- unlike
	// CloudVolumeDownsampleNode's one-time bake, TerrainBiomeNode's weather sim is continuously
	// simulated, so there's no "only regenerate when forced" gate here. Ordered after
	// TerrainBiomeNode via the normal Read<TerrainWeatherBiomeTexture>() accessor edge.
	struct CloudWeatherMinMaxNode: render::NodeRegistrar<CloudWeatherMinMaxNode> {
		static constexpr std::uint32_t kDim = constants::Weather::kCoverageTextureDim;
		static constexpr std::uint32_t kMips = 11; // matches CloudWeatherMinMaxDesc()

		render::PipelineLibrary*                   pipelineLibrary = nullptr;
		ComputeShader                              initShader;       // shaders/cloud_weather_minmax_init.comp
		ComputeShader                              downsampleShader; // shaders/cloud_weather_minmax_downsample.comp
		CloudWeatherMinMaxInitPushConstants         initPush{};
		CloudWeatherMinMaxDownsamplePushConstants   downsamplePush{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!initShader.CompileComputeFromFile(services.device, "shaders/cloud_weather_minmax_init.comp") ||
			    !downsampleShader.CompileComputeFromFile(
					services.device, "shaders/cloud_weather_minmax_downsample.comp"
				)) {
				spdlog::critical("CloudWeatherMinMaxNode shader compilation failed.");
				throw std::runtime_error("CloudWeatherMinMaxNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&initShader);
				services.shaderWatcher->RegisterShader(&downsampleShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<CloudWeatherMinMaxMap>();
			}
		}

		void Destroy(vk::Device device) {
			initShader.Destroy(device);
			downsampleShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<TerrainWeatherBiomeTexture>();
			r.Modify<CloudWeatherMinMaxMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			vk::MemoryBarrier2 barrier(
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
			);
			vk::DependencyInfo dep({}, 1, &barrier);

			initPush.weatherIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			initPush.dstStorageIdx = ctx.StorageIndex<CloudWeatherMinMaxMap>(0);
			initPush.dim = kDim;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, initShader, initPush);
			if (!resolved.pipeline) {
				return;
			}
			std::uint32_t initGroups = (kDim + 15) / 16;
			vkCmd.dispatch(initGroups, initGroups, 1);
			vkCmd.pipelineBarrier2(dep);

			std::uint32_t srcSampledIdx = ctx.Index<CloudWeatherMinMaxMap>();
			for (std::uint32_t mip = 1; mip < kMips; ++mip) {
				std::uint32_t dstDim = (std::max)(1u, kDim >> mip);

				downsamplePush.srcSampledIdx = srcSampledIdx;
				downsamplePush.dstStorageIdx = ctx.StorageIndex<CloudWeatherMinMaxMap>(mip);
				downsamplePush.srcMip = mip - 1;
				downsamplePush.dstDim = dstDim;

				resolved = render::BindCompute(ctx, *pipelineLibrary, downsampleShader, downsamplePush);
				if (!resolved.pipeline) {
					continue;
				}

				std::uint32_t groups = (dstDim + 7) / 8;
				vkCmd.dispatch(groups, groups, 1);

				if (mip + 1 < kMips) {
					vkCmd.pipelineBarrier2(dep);
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(CloudWeatherMinMaxNode);

} // namespace brassica
