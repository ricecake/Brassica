#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct CloudSpatialFilterPushConstants {
		std::uint32_t colorIndex{0};
		std::uint32_t transmittanceIndex{0};
		std::uint32_t momentsIndex{0};
		std::uint32_t depthIndex{0};
		std::uint32_t outColorStorageIdx{0};
		std::uint32_t outTransmittanceStorageIdx{0};
		std::uint32_t stepSize{1};
		std::uint32_t outputWidth{0};
		std::uint32_t outputHeight{0};
	};

	// Stage C: 2 dispatches of the same a-trous shader (step sizes 1 then 2), ping-ponging
	// CloudAccumulatedColor/Transmittance (Stage B's output) -> CloudFilterScratchColor/
	// Transmittance -> CloudFilteredColor/Transmittance (what CloudCompositeNode reads).
	// Hand-rolled barrier between the two dispatches -- same pattern as TonemapNode's bloom-
	// upsample-chain loop and CloudVolumeDownsampleNode's mip chain, since this is one node
	// doing dependent dispatches below the graph's own barrier-synthesis granularity.
	struct CloudSpatialFilterNode: render::NodeRegistrar<CloudSpatialFilterNode> {
		render::PipelineLibrary*       pipelineLibrary = nullptr;
		ComputeShader                  filterShader; // shaders/cloud_spatial_filter.comp
		CloudSpatialFilterPushConstants push{};
		std::uint32_t                   width{0};
		std::uint32_t                   height{0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!filterShader.CompileComputeFromFile(services.device, "shaders/cloud_spatial_filter.comp")) {
				spdlog::critical("CloudSpatialFilterNode shader compilation failed.");
				throw std::runtime_error("CloudSpatialFilterNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&filterShader);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<CloudFilterScratchColor>();
				catalog.Get<CloudFilterScratchTransmittance>();
				catalog.Get<CloudFilteredColor>();
				catalog.Get<CloudFilteredTransmittance>();
			}
		}

		void Destroy(vk::Device device) { filterShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<CloudAccumulatedColor>();
			r.Read<CloudAccumulatedTransmittance>();
			r.Read<CloudMoments>();
			r.Read<CloudDepth>();
			r.Create<CloudFilterScratchColor>();
			r.Create<CloudFilterScratchTransmittance>();
			r.Create<CloudFilteredColor>();
			r.Create<CloudFilteredTransmittance>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::uint32_t momentsIndex = ctx.Index<CloudMoments>();
			std::uint32_t depthIndex = ctx.Index<CloudDepth>();
			std::uint32_t groupsX = (width + 7) / 8;
			std::uint32_t groupsY = (height + 7) / 8;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// Pass 1: CloudAccumulatedColor/Transmittance -> CloudFilterScratchColor/Transmittance
			push.colorIndex = ctx.Index<CloudAccumulatedColor>();
			push.transmittanceIndex = ctx.Index<CloudAccumulatedTransmittance>();
			push.momentsIndex = momentsIndex;
			push.depthIndex = depthIndex;
			push.outColorStorageIdx = ctx.StorageIndex<CloudFilterScratchColor>();
			push.outTransmittanceStorageIdx = ctx.StorageIndex<CloudFilterScratchTransmittance>();
			push.stepSize = 1;
			push.outputWidth = width;
			push.outputHeight = height;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, filterShader, push);
			if (resolved.pipeline) {
				vkCmd.dispatch(groupsX, groupsY, 1);

				vk::MemoryBarrier2 barrier(
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
				);
				vk::DependencyInfo dep({}, 1, &barrier);
				vkCmd.pipelineBarrier2(dep);
			}

			// Pass 2: CloudFilterScratchColor/Transmittance -> CloudFilteredColor/Transmittance
			push.colorIndex = ctx.Index<CloudFilterScratchColor>();
			push.transmittanceIndex = ctx.Index<CloudFilterScratchTransmittance>();
			push.outColorStorageIdx = ctx.StorageIndex<CloudFilteredColor>();
			push.outTransmittanceStorageIdx = ctx.StorageIndex<CloudFilteredTransmittance>();
			push.stepSize = 2;

			resolved = render::BindCompute(ctx, *pipelineLibrary, filterShader, push);
			if (resolved.pipeline) {
				vkCmd.dispatch(groupsX, groupsY, 1);
			}
		}
	};

	BRASSICA_REGISTER_NODE(CloudSpatialFilterNode);

} // namespace brassica
