#pragma once

#include <algorithm>
#include <cmath>
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

	struct CloudBoundingPushConstants {
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t weatherMinMaxIndex{0};
		std::uint32_t dstStorageIdx{0};
		std::uint32_t outputWidth{0};
		std::uint32_t outputHeight{0};
		std::uint32_t scaleFactor{1};
		std::uint32_t frameIndex{0};
	};

	// Stage E1: cheap coarse bounding march (shaders/cloud_bounding.comp), interleaved by
	// FrameContext::renderScalePercent. Writes CloudBoundingMap, which CloudTileSchedulerNode
	// (F3) and CloudTemporalNode (F2) read -- must run before both, established via the normal
	// Read<CloudBoundingMap>() accessor edge on this node's sole Modify<>(), no explicit
	// Before<>()/After<>() needed, same reasoning as the rest of this pipeline.
	struct CloudBoundingNode: render::NodeRegistrar<CloudBoundingNode> {
		render::PipelineLibrary*    pipelineLibrary = nullptr;
		ComputeShader               boundingShader; // shaders/cloud_bounding.comp
		CloudBoundingPushConstants  push{};
		std::uint32_t               width{0};
		std::uint32_t               height{0};
		std::uint32_t               scaleFactor{1};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!boundingShader.CompileComputeFromFile(services.device, "shaders/cloud_bounding.comp")) {
				spdlog::critical("CloudBoundingNode shader compilation failed.");
				throw std::runtime_error("CloudBoundingNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&boundingShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<CloudBoundingMap>();
			}
		}

		void Destroy(vk::Device device) { boundingShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;
			// round(1 / max(0.001, renderScalePercent/100)) -- degenerates to 1 at the default
			// 100%, matching boidish's own uRenderScale-fraction formula exactly.
			float renderScaleFraction = static_cast<float>(ctx.renderScalePercent) / 100.0f;
			scaleFactor =
				(std::max)(1u, static_cast<std::uint32_t>(std::lround(1.0f / (std::max)(0.001f, renderScaleFraction))));

			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<CloudWeatherMinMaxMap>();
			r.Modify<CloudBoundingMap>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.weatherMinMaxIndex = ctx.Index<CloudWeatherMinMaxMap>();
			push.dstStorageIdx = ctx.StorageIndex<CloudBoundingMap>();
			push.outputWidth = width;
			push.outputHeight = height;
			push.scaleFactor = scaleFactor;
			push.frameIndex = static_cast<std::uint32_t>(ctx.frameIndex);

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, boundingShader, push);
			if (!resolved.pipeline) {
				return;
			}

			std::uint32_t fullResGroupsX = (width + 7) / 8;
			std::uint32_t fullResGroupsY = (height + 7) / 8;
			std::uint32_t dispatchGroupsX = (fullResGroupsX + scaleFactor - 1) / scaleFactor;
			std::uint32_t dispatchGroupsY = (fullResGroupsY + scaleFactor - 1) / scaleFactor;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(dispatchGroupsX, dispatchGroupsY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudBoundingNode);

} // namespace brassica
