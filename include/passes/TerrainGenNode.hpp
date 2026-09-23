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

	struct QuadtreePushConstants {
		std::uint32_t biomeTextureIndex{0};
		std::uint32_t minMaxTextureIndex{0};
		float         planetRadius{600000.0f};
		std::uint32_t maxTreeDepth{10};
	};

	struct MeshletGenPushConstants {
		std::uint32_t clipmapIndex{0};
		std::uint32_t biomeIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t visibilityIndex{0};
	};

	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		using Resources = graph::Declares<
			graph::Modify<TerrainClipmapTexture>,
			graph::Modify<TerrainMinMaxTexture>,
			graph::Modify<TerrainBiomeTexture>,
			graph::Modify<TerrainTileVisibilityTexture>,
			graph::Modify<TerrainTLAS>,
			graph::Create<TerrainLowResChunkTexture>,
			graph::Create<TerrainQuadtreeBuffer>,
			graph::Create<TerrainPageTableBuffer>,
			graph::Create<TerrainVertexPageBuffer>,
			graph::Create<TerrainNodeCreateBuffer>,
			graph::Create<TerrainFreePagePoolBuffer>>;

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
		ComputeShader                    quadtreeShader;
		ComputeShader                    meshletGenShader;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainGenPushConstants          push{};
		glm::vec3                        cameraPos{0.0f};
		bool                             hasUpdate{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			physicalRegistry = services.physicalRegistry;
			genShader.CompileComputeFromFile(services.device, "shaders/terrain_gen.comp");
			aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp");
			quadtreeShader.CompileComputeFromFile(services.device, "shaders/terrain_quadtree.comp");
			meshletGenShader.CompileComputeFromFile(services.device, "shaders/terrain_meshlet_gen.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&aabbShader);
			watcher.RegisterShader(&quadtreeShader);
			watcher.RegisterShader(&meshletGenShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
			quadtreeShader.Destroy(device);
			meshletGenShader.Destroy(device);
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
			r.realizations.reserve(10);
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
					.key = graph::IdOf<TerrainTLAS>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::AccelerationStructureDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainLowResChunkTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainLowResChunkDesc(push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainQuadtreeBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(1024 * 64),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainPageTableBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(4096 * 32),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainVertexPageBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(2048 * 121 * 48),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainNodeCreateBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(2048 * 32),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainFreePagePoolBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(2048 * 4 + 16),
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

			// Quadtree Traversal & Leaf Splitting/Merging
			{
				QuadtreePushConstants qtPush{
					.biomeTextureIndex = push.biomeStorageIdx,
					.minMaxTextureIndex = push.minMaxStorageIdx,
					.planetRadius = 600000.0f,
					.maxTreeDepth = push.gridParams.x,
				};
				std::array<vk::PushConstantRange, 1> qtPushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(QuadtreePushConstants)}
				};
				render::ComputePipelineRequest qtRequest{
					.shader = &quadtreeShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = qtPushRanges,
				};
				render::ResolvedPipeline qtResolved = pipelineLibrary->ResolveCached(qtRequest);
				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (qtResolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, qtResolved.pipeline);
					if (boundSets[0] && boundSets[1]) {
						vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, qtResolved.layout, 0, boundSets, nullptr);
					}
					vkCmd.pushConstants(
						qtResolved.layout,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sizeof(QuadtreePushConstants),
						&qtPush
					);
					vkCmd.dispatch(16, 1, 1);
				}
			}

			// Indirect Meshlet Vertex Generation
			{
				MeshletGenPushConstants mgPush{
					.clipmapIndex = push.clipmapStorageIdx,
					.biomeIndex = push.biomeStorageIdx,
					.minMaxIndex = push.minMaxStorageIdx,
					.visibilityIndex = push.visibilityStorageIdx,
				};
				std::array<vk::PushConstantRange, 1> mgPushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MeshletGenPushConstants)}
				};
				render::ComputePipelineRequest mgRequest{
					.shader = &meshletGenShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = mgPushRanges,
				};
				render::ResolvedPipeline mgResolved = pipelineLibrary->ResolveCached(mgRequest);
				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (mgResolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, mgResolved.pipeline);
					if (boundSets[0] && boundSets[1]) {
						vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, mgResolved.layout, 0, boundSets, nullptr);
					}
					vkCmd.pushConstants(
						mgResolved.layout,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sizeof(MeshletGenPushConstants),
						&mgPush
					);
					if (physicalRegistry) {
						if (auto createBuf = physicalRegistry->GetBuffer<TerrainNodeCreateBuffer>()) {
							vkCmd.dispatchIndirect(createBuf->GetBuffer(), 0);
						} else {
							vkCmd.dispatch(1, 1, 1);
						}
					} else {
						vkCmd.dispatch(1, 1, 1);
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
