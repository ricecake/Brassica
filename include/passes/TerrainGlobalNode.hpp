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

	struct TerrainGlobalGenPushConstants {
		std::uint32_t globalStorageIdx{0};
		std::uint32_t globalMinMaxStorageIdx{0};
		std::uint32_t globalBiomeStorageIdx{0};
		std::uint32_t dim{4192};
	};

	struct TerrainGlobalDownsamplePushConstants {
		std::uint32_t srcMinMaxStorageIdx{0};
		std::uint32_t dstMinMaxStorageIdx{0};
		std::uint32_t dstDim{2096};
		std::uint32_t srcDim{4192};
		std::uint32_t srcMip{0};
	};

	struct TerrainGlobalNode: render::NodeRegistrar<TerrainGlobalNode> {
		render::PipelineLibrary*             pipelineLibrary = nullptr;
		ComputeShader                        genShader;
		ComputeShader                        downsampleShader;
		TerrainGlobalGenPushConstants        genPush{};
		TerrainGlobalDownsamplePushConstants downPush{};
		bool                                 m_generated = false;
		bool                                 forceRegeneration = true;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_global_gen.comp") ||
			    !downsampleShader.CompileComputeFromFile(services.device, "shaders/terrain_global_downsample.comp")) {
				spdlog::critical("TerrainGlobalNode shader compilation failed.");
				throw std::runtime_error("TerrainGlobalNode shader compilation failed.");
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
			r.Create<TerrainGlobalTexture>();
			r.Create<TerrainGlobalMinMaxTexture>();
			r.Create<TerrainGlobalBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (m_generated && !forceRegeneration) {
				return;
			}

			genPush.globalStorageIdx = ctx.StorageIndex<TerrainGlobalTexture>();
			genPush.globalMinMaxStorageIdx = ctx.StorageIndex<TerrainGlobalMinMaxTexture>(0);
			genPush.globalBiomeStorageIdx = ctx.StorageIndex<TerrainGlobalBiomeTexture>();
			genPush.dim = 4192;

			render::ResolvedPipeline resolvedGen = render::BindCompute(ctx, *pipelineLibrary, genShader, genPush);
			if (!resolvedGen.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			uint32_t          groupDim = (4192 + 15) / 16;
			vkCmd.dispatch(groupDim, groupDim, 1);

			// Downsample min-max mips 1..12
			std::uint32_t srcDim = 4192;
			for (std::uint32_t mip = 1; mip < 13; ++mip) {
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

				downPush.srcMinMaxStorageIdx = ctx.Index<TerrainGlobalMinMaxTexture>();
				downPush.dstMinMaxStorageIdx = ctx.StorageIndex<TerrainGlobalMinMaxTexture>(mip);
				downPush.dstDim = dstDim;
				downPush.srcDim = srcDim;
				downPush.srcMip = mip - 1;

				render::ResolvedPipeline resolvedDown =
					render::BindCompute(ctx, *pipelineLibrary, downsampleShader, downPush);
				if (resolvedDown.pipeline) {
					uint32_t dstGroupDim = (dstDim + 15) / 16;
					vkCmd.dispatch(dstGroupDim, dstGroupDim, 1);
				}

				srcDim = dstDim;
			}

			m_generated = true;
			forceRegeneration = false;
		}
	};

	BRASSICA_REGISTER_NODE(TerrainGlobalNode);

} // namespace brassica
