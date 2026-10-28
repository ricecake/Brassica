#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
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
#include "terrain/TerrainAccelerationStructure.hpp"

namespace brassica {

	class ShaderWatcher;

	struct TerrainPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim (offset 0)
		std::uint32_t clipmapIndex{0};       // offset 16
		std::uint32_t minMaxIndex{0};        // offset 20
		std::uint32_t biomeIndex{0};         // offset 24
		float         waterLevel{0.0f};      // offset 28
		std::uint32_t weatherBiomeIndex{0}; // offset 32
		std::uint32_t cbtBitfieldIndex{0};  // offset 36
		std::uint32_t cbtPoolIndex{0};      // offset 40
		std::uint32_t cbtCommandsIndex{0};  // offset 44
		std::uint32_t cbtPageTableIndex{0}; // offset 48
		std::uint32_t cbtCacheIndex{0};     // offset 52
		std::uint32_t activeBisectorCount{2}; // offset 56
		std::uint32_t _pad0{0};             // offset 60
	};

	static_assert(sizeof(TerrainPushConstants) == 64, "TerrainPushConstants size must be 64 bytes");

	struct TerrainNode: render::NodeRegistrar<TerrainNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*      pipelineLibrary = nullptr;
		TaskShader                    taskShader;
		MeshShader                    meshShader;
		FragmentShader                fragShader;
		ComputeShader                 cbtResetCounterComp;
		ComputeShader                 cbtCachePointersComp;
		ComputeShader                 cbtResetCommandsComp;
		ComputeShader                 cbtGenerateCommandsComp;
		ComputeShader                 cbtReserveBlocksComp;
		ComputeShader                 cbtFillBlocksComp;
		ComputeShader                 cbtUpdateNeighborsComp;
		ComputeShader                 cbtUpdateBitfieldComp;
		ComputeShader                 cbtSumReductionComp;
		ComputeShader                 cbtTerrainBakeComp;
		TerrainAccelerationStructure* terrainAS = nullptr;
		TerrainPushConstants          push{};
		std::uint32_t                 m_activeBisectorCount{2};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/terrain.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/terrain.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/terrain.frag") ||
			    !cbtResetCounterComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_reset_counter.comp") ||
			    !cbtCachePointersComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_cache_pointers.comp") ||
			    !cbtResetCommandsComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_reset_commands.comp") ||
			    !cbtGenerateCommandsComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_generate_commands.comp") ||
			    !cbtReserveBlocksComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_reserve_blocks.comp") ||
			    !cbtFillBlocksComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_fill_blocks.comp") ||
			    !cbtUpdateNeighborsComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_update_neighbors.comp") ||
			    !cbtUpdateBitfieldComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_update_bitfield.comp") ||
			    !cbtSumReductionComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_sum_reduction.comp") ||
			    !cbtTerrainBakeComp.CompileComputeFromFile(services.device, "shaders/terrain/cbt_terrain_bake.comp")) {
				spdlog::critical("TerrainNode shader compilation failed.");
				throw std::runtime_error("TerrainNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<GBufferPosition>();
				catalog.Get<GBufferNormal>();
				catalog.Get<GBufferAlbedo>();
				catalog.Get<GBufferMaterial>();
				catalog.Get<GBufferDepth>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
			watcher.RegisterShader(&cbtResetCounterComp);
			watcher.RegisterShader(&cbtCachePointersComp);
			watcher.RegisterShader(&cbtResetCommandsComp);
			watcher.RegisterShader(&cbtGenerateCommandsComp);
			watcher.RegisterShader(&cbtReserveBlocksComp);
			watcher.RegisterShader(&cbtFillBlocksComp);
			watcher.RegisterShader(&cbtUpdateNeighborsComp);
			watcher.RegisterShader(&cbtUpdateBitfieldComp);
			watcher.RegisterShader(&cbtSumReductionComp);
			watcher.RegisterShader(&cbtTerrainBakeComp);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
			cbtResetCounterComp.Destroy(device);
			cbtCachePointersComp.Destroy(device);
			cbtResetCommandsComp.Destroy(device);
			cbtGenerateCommandsComp.Destroy(device);
			cbtReserveBlocksComp.Destroy(device);
			cbtFillBlocksComp.Destroy(device);
			cbtUpdateNeighborsComp.Destroy(device);
			cbtUpdateBitfieldComp.Destroy(device);
			cbtSumReductionComp.Destroy(device);
			cbtTerrainBakeComp.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.gridParams = p.terrainGridParams;
			push.waterLevel = p.waterLevel;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Create<GBufferPosition>().Clear({0.0f, 0.0f, 0.0f, 0.0f});
			r.Create<GBufferNormal>().Clear({0.0f, 0.0f, 0.0f, 0.0f});
			r.Create<GBufferAlbedo>().Clear({0.0f, 0.0f, 0.0f, 0.0f});
			r.Create<GBufferMaterial>().Clear({0.0f, 0.0f, 0.0f, 0.0f});
			r.Create<GBufferDepth>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<TerrainBiomeTexture>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Create<CBTBitfieldBuffer>();
			r.Create<CBTBisectorPoolBuffer>().Clear({0.0f, 0.0f, 0.0f, 0.0f});
			r.Create<CBTCommandsBuffer>();
			r.Create<CBTPageTableBuffer>();
			r.Create<CBTGeometryCacheBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.cbtBitfieldIndex = ctx.BufferIndex<CBTBitfieldBuffer>();
			push.cbtPoolIndex = ctx.BufferIndex<CBTBisectorPoolBuffer>();
			push.cbtCommandsIndex = ctx.BufferIndex<CBTCommandsBuffer>();
			push.cbtPageTableIndex = ctx.BufferIndex<CBTPageTableBuffer>();
			push.cbtCacheIndex = ctx.BufferIndex<CBTGeometryCacheBuffer>();
			push.activeBisectorCount = std::clamp(m_activeBisectorCount, 2u, 131072u);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			vk::MemoryBarrier computeBarrier{
				vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
				vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead
			};

			uint32_t computeGroups = (push.activeBisectorCount + 63) / 64;

			render::BindCompute(ctx, *pipelineLibrary, cbtResetCounterComp, push);
			vkCmd.dispatch(1, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtCachePointersComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtResetCommandsComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtGenerateCommandsComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtReserveBlocksComp, push);
			vkCmd.dispatch(1, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtFillBlocksComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtUpdateNeighborsComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtUpdateBitfieldComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtSumReductionComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			render::BindCompute(ctx, *pipelineLibrary, cbtTerrainBakeComp, push);
			vkCmd.dispatch(computeGroups, 1, 1);
			vkCmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eTaskShaderEXT | vk::PipelineStageFlagBits::eMeshShaderEXT,
				vk::DependencyFlags{},
				computeBarrier,
				{},
				{}
			);

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb,
                vk::Format::eR8G8B8A8Unorm
            };
			render::ResolvedPipeline resolved = render::BindGraphics(
				ctx,
				*pipelineLibrary,
				stages,
				kPipelineState,
				colorFormats,
				push,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
					vk::ShaderStageFlagBits::eFragment,
				vk::Format::eD32Sfloat
			);
			if (!resolved.pipeline) {
				return;
			}

			uint32_t taskGroupCount = (push.activeBisectorCount + 31) / 32;
			if (terrainAS && terrainAS->GetDls().vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, terrainAS->GetDls());
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainNode);

} // namespace brassica
