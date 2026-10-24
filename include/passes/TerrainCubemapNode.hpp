#pragma once

#include <algorithm>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct TerrainCubemapGenPushConstants {
		std::uint32_t cubemapStorageIdx{0};
		std::uint32_t cubemapMinMaxStorageIdx{0};
		std::uint32_t cubemapBiomeStorageIdx{0};
		std::uint32_t dim{8192};
	};

	struct TerrainCubemapDownsamplePushConstants {
		std::uint32_t srcMinMaxStorageIdx{0};
		std::uint32_t dstMinMaxStorageIdx{0};
		std::uint32_t dstDim{4096};
		std::uint32_t srcDim{8192};
		std::uint32_t srcMip{0};
	};

	struct TerrainCubemapNode: render::NodeRegistrar<TerrainCubemapNode> {
		render::PipelineLibrary*              pipelineLibrary = nullptr;
		ComputeShader                         genShader;
		ComputeShader                         downsampleShader;
		TerrainCubemapGenPushConstants        genPush{};
		TerrainCubemapDownsamplePushConstants downPush{};
		bool                                  m_generated = false;
		bool                                  forceRegeneration = true;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_cubemap_gen.comp") ||
			    !downsampleShader.CompileComputeFromFile(services.device, "shaders/terrain_cubemap_downsample.comp")) {
				spdlog::critical("TerrainCubemapNode shader compilation failed.");
				throw std::runtime_error("TerrainCubemapNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&downsampleShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			downsampleShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			if (p.forceRegeneration) {
				forceRegeneration = true;
			}
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainCubemapTexture>();
			r.Create<TerrainCubemapMinMaxTexture>();
			r.Create<TerrainCubemapBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (m_generated && !forceRegeneration) {
				return;
			}

			genPush.cubemapStorageIdx = ctx.StorageIndex<TerrainCubemapTexture>();
			genPush.cubemapMinMaxStorageIdx = ctx.StorageIndex<TerrainCubemapMinMaxTexture>(0);
			genPush.cubemapBiomeStorageIdx = ctx.StorageIndex<TerrainCubemapBiomeTexture>();
			genPush.dim = 8192;

			render::ResolvedPipeline resolvedGen = render::BindCompute(ctx, *pipelineLibrary, genShader, genPush);
			if (!resolvedGen.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			uint32_t          groupDim = (8192 + 15) / 16;
			vkCmd.dispatch(groupDim, groupDim, 6);

			// Downsample min-max mips 1..13
			std::uint32_t srcDim = 8192;
			for (std::uint32_t mip = 1; mip < 14; ++mip) {
				std::uint32_t dstDim = std::max(1u, srcDim >> 1);

				// Memory barrier between mips
				vk::MemoryBarrier memoryBarrier(
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead
				);
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					1,
					&memoryBarrier,
					0,
					nullptr,
					0,
					nullptr
				);

				downPush.srcMinMaxStorageIdx = ctx.Index<TerrainCubemapMinMaxTexture>();
				downPush.dstMinMaxStorageIdx = ctx.StorageIndex<TerrainCubemapMinMaxTexture>(mip);
				downPush.dstDim = dstDim;
				downPush.srcDim = srcDim;
				downPush.srcMip = mip - 1;

				render::ResolvedPipeline resolvedDown =
					render::BindCompute(ctx, *pipelineLibrary, downsampleShader, downPush);
				if (resolvedDown.pipeline) {
					uint32_t dstGroupDim = (dstDim + 15) / 16;
					vkCmd.dispatch(dstGroupDim, dstGroupDim, 6);
				}

				srcDim = dstDim;
			}

			m_generated = true;
			forceRegeneration = false;
		}
	};

	BRASSICA_REGISTER_NODE(TerrainCubemapNode);

} // namespace brassica
