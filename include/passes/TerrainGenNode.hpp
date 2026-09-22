#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"
#include "terrain/TerrainClipmap.hpp"

namespace brassica {

	struct TerrainGenPushConstants {
		glm::uvec4 gridParams{10, 16, 2560, 1088}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::uvec4 lodOffsets0_3{0u};
		glm::uvec4 lodOffsets4_7{0u};
		glm::uvec4 lodOffsets8_11{0u};
		glm::uvec4 lodDeltas0_3{0u};
		glm::uvec4 lodDeltas4_7{0u};
		glm::uvec4 lodDeltas8_11{0u};
		std::uint32_t clipmapStorageIdx{0};
		std::uint32_t minMaxStorageIdx{0};
		std::uint32_t biomeStorageIdx{0};
		std::uint32_t visibilityStorageIdx{0};
	};

	struct TerrainChunkGenPushConstants {
		glm::vec2     minWorldPos{0.0f, 0.0f};
		std::uint32_t chunkSlot{0};
		std::uint32_t chunkStorageIdx{0};
		float         texelSize{TERRAIN_CHUNK_TEXEL_SIZE};
	};

	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		using Resources = graph::Declares<
			graph::Modify<TerrainClipmapTexture>,
			graph::Modify<TerrainMinMaxTexture>,
			graph::Modify<TerrainBiomeTexture>,
			graph::Modify<TerrainTileVisibilityTexture>,
			graph::Modify<TerrainChunkTexture>,
			graph::Modify<TerrainTLAS>>;

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		TerrainClipmap*                  terrainClipmap = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
		ComputeShader                    chunkGenShader;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainGenPushConstants          push{};
		glm::vec3                        cameraPos{0.0f};
		bool                             hasUpdate{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			terrainClipmap = services.terrainClipmap;
			physicalRegistry = services.physicalRegistry;
			genShader.CompileComputeFromFile(services.device, "shaders/terrain_gen.comp");
			aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp");
			chunkGenShader.CompileComputeFromFile(services.device, "shaders/terrain_chunk_gen.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&aabbShader);
			watcher.RegisterShader(&chunkGenShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
			chunkGenShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.gridParams = p.terrainGridParams;
			push.lodOffsets0_3 = p.terrainLodOffsets0_3;
			push.lodOffsets4_7 = p.terrainLodOffsets4_7;
			push.lodOffsets8_11 = p.terrainLodOffsets8_11;
			push.lodDeltas0_3 = p.terrainLodDeltas0_3;
			push.lodDeltas4_7 = p.terrainLodDeltas4_7;
			push.lodDeltas8_11 = p.terrainLodDeltas8_11;
			hasUpdate = p.terrainHasUpdate;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(6);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainClipmapTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainClipmapDesc(push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxDesc(push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainBiomeTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainBiomeDesc(push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainTileVisibilityTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainTileVisibilityDesc(push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainChunkTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainChunkDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainTLAS>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::AccelerationStructureDesc(),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout))
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet))
			};

			if (hasUpdate) {
				push.clipmapStorageIdx = ctx.StorageIndex<TerrainClipmapTexture>();
				push.minMaxStorageIdx = ctx.StorageIndex<TerrainMinMaxTexture>();
				push.biomeStorageIdx = ctx.StorageIndex<TerrainBiomeTexture>();
				push.visibilityStorageIdx = ctx.StorageIndex<TerrainTileVisibilityTexture>();

				std::array<vk::PushConstantRange, 1> pushConstantRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainGenPushConstants)}
				};

				render::ComputePipelineRequest request{
					.shader = &genShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = pushConstantRanges,
				};
				render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (resolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
				}

				if (boundSets[0] && boundSets[1]) {
					vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
				}

				vkCmd.pushConstants(
					resolved.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(TerrainGenPushConstants),
					&push
				);

				uint32_t groupX = (push.gridParams.w + 15) / 16;
				uint32_t groupY = (push.gridParams.w + 15) / 16;
				uint32_t groupZ = push.gridParams.x;
				vkCmd.dispatch(groupX, groupY, groupZ);
			}

			if (terrainClipmap) {
				uint32_t chunkStorageIdx = ctx.StorageIndex<TerrainChunkTexture>();
				const auto& chunks = terrainClipmap->GetChunkInfos();

				std::array<vk::PushConstantRange, 1> pushConstantRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainChunkGenPushConstants)}
				};

				render::ComputePipelineRequest request{
					.shader = &chunkGenShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = pushConstantRanges,
				};
				render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (resolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
				}

				if (boundSets[0] && boundSets[1]) {
					vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
				}

				for (const auto& chunk : chunks) {
					if (chunk.isDirty) {
						TerrainChunkGenPushConstants chunkPush{
							.minWorldPos = chunk.minWorldPos,
							.chunkSlot = chunk.slot,
							.chunkStorageIdx = chunkStorageIdx,
							.texelSize = TERRAIN_CHUNK_TEXEL_SIZE
						};

						vkCmd.pushConstants(
							resolved.layout,
							vk::ShaderStageFlagBits::eCompute,
							0,
							sizeof(TerrainChunkGenPushConstants),
							&chunkPush
						);

						vkCmd.dispatch((TERRAIN_CHUNK_DIM + 15) / 16, (TERRAIN_CHUNK_DIM + 15) / 16, 1);
					}
				}
			}

			if (terrainAS) {
				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				terrainAS->BuildOrUpdate(
					vkCmd,
					cameraPos,
					0.5f,
					push.gridParams.x,
					pipelineLibrary,
					&aabbShader,
					boundSets[0],
					boundSets[1],
					setLayouts[0],
					setLayouts[1],
					push.gridParams,
					push.lodOffsets0_3,
					push.lodOffsets4_7,
					push.lodOffsets8_11
				);
				if (physicalRegistry && terrainAS->GetTLAS()) {
					physicalRegistry->RegisterImportedAccelerationStructure<TerrainTLAS>(terrainAS->GetTLAS());
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainGenNode);

} // namespace brassica
