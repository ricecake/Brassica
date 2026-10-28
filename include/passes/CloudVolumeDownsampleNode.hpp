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
	struct CloudVolumeGenNode;

	struct CloudVolumeDownsamplePushConstants {
		std::uint32_t srcSampledIdx{0};
		std::uint32_t dstStorageIdx{0};
		std::uint32_t srcMip{0};
		std::uint32_t dstDim{0};
	};

	// Builds CloudDetailVolume's mip chain (mips 1-4, CloudVolumeDesc()) from mip 0, which
	// CloudVolumeGenNode writes. A separate node, not folded into CloudVolumeGenNode::Execute,
	// specifically so the barrier between "write mip 0" and "read mip 0 / write mip 1" is
	// synthesized by the frame graph itself instead of hand-rolled -- same reasoning as
	// TerrainMinMaxDownsampleNode's identical split from TerrainGenNode (Modify<>() ordered
	// after CloudVolumeGenNode's sole Create<>() via Create-before-accessor, no explicit
	// After<>() needed). The barriers *between* this node's own mip-to-mip dispatches (mip 1 ->
	// mip 2, etc.) still have to be hand-rolled -- TonemapNode's bloom-upsample-chain loop is the
	// precedent -- since those are within one node's Execute(), below the graph's own barrier-
	// synthesis granularity.
	//
	// Exists to fix the ~10x GPU regression alligator() noise caused in cloud_volume_gen.comp
	// (see that file + cloud_render.frag's sampleCloudVolume): raw unfiltered sharp Worley-cell
	// edges made neighboring raymarch threads disagree on threshold branches. A real mip chain
	// lets distant samples read pre-blurred data instead.
	struct CloudVolumeDownsampleNode: render::NodeRegistrar<CloudVolumeDownsampleNode> {
		static constexpr std::uint32_t kVolumeDim = 256;
		static constexpr std::uint32_t kVolumeMips = 5; // mips 0..4 -- matches CloudVolumeDesc()

		render::PipelineLibrary*           pipelineLibrary = nullptr;
		ComputeShader                      downsampleShader;
		CloudVolumeDownsamplePushConstants push{};
		bool                               hasDownsampled{false};
		bool                               forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!downsampleShader.CompileComputeFromFile(services.device, "shaders/cloud_volume_downsample.comp")) {
				spdlog::critical("CloudVolumeDownsampleNode shader compilation failed.");
				throw std::runtime_error("CloudVolumeDownsampleNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&downsampleShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<CloudDetailVolume>();
			}
		}

		void Destroy(vk::Device device) { downsampleShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { forceRegeneration = p.forceRegeneration; }

		// CloudDetailVolume is catalog-converted -- no .Desc() needed, same as
		// TerrainMinMaxDownsampleNode's identical note on TerrainMinMaxTexture.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(!hasDownsampled || forceRegeneration);
			r.Modify<CloudDetailVolume>();
			r.After<CloudVolumeGenNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::uint32_t srcSampledIdx = ctx.Index<CloudDetailVolume>();
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			for (std::uint32_t mip = 1; mip < kVolumeMips; ++mip) {
				std::uint32_t dstDim = (std::max)(1u, kVolumeDim >> mip);

				push.srcSampledIdx = srcSampledIdx;
				push.dstStorageIdx = ctx.StorageIndex<CloudDetailVolume>(mip);
				push.srcMip = mip - 1;
				push.dstDim = dstDim;

				render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, downsampleShader, push);
				if (!resolved.pipeline) {
					continue;
				}

				std::uint32_t groups = (dstDim + 7) / 8;
				vkCmd.dispatch(groups, groups, groups);

				if (mip + 1 < kVolumeMips) {
					vk::MemoryBarrier2 barrier(
						vk::PipelineStageFlagBits2::eComputeShader,
						vk::AccessFlagBits2::eShaderStorageWrite,
						vk::PipelineStageFlagBits2::eComputeShader,
						vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
					);
					vk::DependencyInfo dep({}, 1, &barrier);
					vkCmd.pipelineBarrier2(dep);
				}
			}

			hasDownsampled = true;
		}
	};

	BRASSICA_REGISTER_NODE(CloudVolumeDownsampleNode);

} // namespace brassica
