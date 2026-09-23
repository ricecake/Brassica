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
		std::uint32_t lowResChunkStorageIdx{0};
	};

	struct QuadtreePushConstants {
		std::uint64_t quadtreeAddr{0};
		std::uint64_t freePoolAddr{0};
		std::uint64_t createTasksAddr{0};
		std::uint64_t pageTableAddr{0};
		std::uint32_t biomeTextureIndex{0};
		std::uint32_t minMaxTextureIndex{0};
		std::uint32_t lowResChunkTextureIndex{0};
		float         planetRadius{600000.0f};
		std::uint32_t maxTreeDepth{10};
	};

	struct MeshletGenPushConstants {
		std::uint64_t createTasksAddr{0};
		std::uint64_t vertexPagePoolAddr{0};
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
		bool                             freePoolInitialized{false};
		bool                             quadtreeInitialized{false};

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
					.desc = graph::StorageBufferDesc(16 + 4096 * 64),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainPageTableBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(16 + 4096 * 32),
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
					.desc = graph::StorageBufferDesc(16 + 2048 * 32),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainFreePagePoolBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(16 + 2048 * 4),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			if (physicalRegistry) {
				if (auto ptBuf = physicalRegistry->GetBuffer<TerrainPageTableBuffer>()) {
					vkCmd.fillBuffer(ptBuf->GetBuffer(), 0, 16, 0);
				}
				if (auto createBuf = physicalRegistry->GetBuffer<TerrainNodeCreateBuffer>()) {
					struct IndirectInit { std::uint32_t x{0}; std::uint32_t y{1}; std::uint32_t z{1}; std::uint32_t taskCount{0}; };
					IndirectInit initCmd{};
					vkCmd.updateBuffer(createBuf->GetBuffer(), 0, sizeof(IndirectInit), &initCmd);
				}
				if (auto poolBuf = physicalRegistry->GetBuffer<TerrainFreePagePoolBuffer>()) {
					if (!freePoolInitialized) {
						struct PagePoolHeader { std::uint32_t freeCount{2048}; std::uint32_t capacity{2048}; std::uint32_t allocatedCount{0}; std::uint32_t padding{0}; };
						PagePoolHeader hdr{};
						vkCmd.updateBuffer(poolBuf->GetBuffer(), 0, sizeof(PagePoolHeader), &hdr);
						std::vector<std::uint32_t> initialPages(2048);
						for (std::uint32_t i = 0; i < 2048; ++i) initialPages[i] = i;
						vkCmd.updateBuffer(poolBuf->GetBuffer(), sizeof(PagePoolHeader), initialPages.size() * sizeof(std::uint32_t), initialPages.data());
						freePoolInitialized = true;
					}
				}
				if (auto qtBuf = physicalRegistry->GetBuffer<TerrainQuadtreeBuffer>()) {
					if (!quadtreeInitialized) {
						struct QuadtreeHeader { std::uint32_t nodeCount{1}; std::uint32_t maxNodes{1024}; std::uint32_t leafCount{1}; std::uint32_t padding{0}; };
						QuadtreeHeader hdr{};
						vkCmd.updateBuffer(qtBuf->GetBuffer(), 0, sizeof(QuadtreeHeader), &hdr);

						struct QuadtreeNode {
							glm::vec4 bounds{-65536.0f, -65536.0f, 131072.0f, 131072.0f};
							std::uint32_t lod{10};
							std::uint32_t pageIndex{0xFFFFFFFFu};
							std::uint32_t flags{3u};
							std::uint32_t padding{0};
							glm::uvec4 children{0u};
						};
						QuadtreeNode rootNode{};
						vkCmd.updateBuffer(qtBuf->GetBuffer(), sizeof(QuadtreeHeader), sizeof(QuadtreeNode), &rootNode);
						quadtreeInitialized = true;
					}
				}

				vk::MemoryBarrier2 transferBarrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
					.srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
				};
				vk::DependencyInfo transferDepInfo{};
				transferDepInfo.setMemoryBarriers(transferBarrier);
				vkCmd.pipelineBarrier2(transferDepInfo);
			}

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
				push.lowResChunkStorageIdx = ctx.StorageIndex<TerrainLowResChunkTexture>();

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
				std::uint64_t qtAddr = 0, poolAddr = 0, createAddr = 0, ptAddr = 0;
				if (physicalRegistry) {
					if (auto buf = physicalRegistry->GetBuffer<TerrainQuadtreeBuffer>()) qtAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
					if (auto buf = physicalRegistry->GetBuffer<TerrainFreePagePoolBuffer>()) poolAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
					if (auto buf = physicalRegistry->GetBuffer<TerrainNodeCreateBuffer>()) createAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
					if (auto buf = physicalRegistry->GetBuffer<TerrainPageTableBuffer>()) ptAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
				}

				QuadtreePushConstants qtPush{
					.quadtreeAddr = qtAddr,
					.freePoolAddr = poolAddr,
					.createTasksAddr = createAddr,
					.pageTableAddr = ptAddr,
					.biomeTextureIndex = push.biomeStorageIdx,
					.minMaxTextureIndex = push.minMaxStorageIdx,
					.lowResChunkTextureIndex = push.lowResChunkStorageIdx,
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

			// Compute Memory Barrier between Quadtree Traversal and Indirect Vertex Generation
			{
				vk::MemoryBarrier2 barrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eIndirectCommandRead,
				};
				vk::DependencyInfo depInfo{};
				depInfo.setMemoryBarriers(barrier);
				vkCmd.pipelineBarrier2(depInfo);
			}

			// Indirect Meshlet Vertex Generation
			{
				std::uint64_t createAddr = 0, vertPoolAddr = 0;
				if (physicalRegistry) {
					if (auto buf = physicalRegistry->GetBuffer<TerrainNodeCreateBuffer>()) createAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
					if (auto buf = physicalRegistry->GetBuffer<TerrainVertexPageBuffer>()) vertPoolAddr = static_cast<std::uint64_t>(buf->GetDeviceAddress());
				}

				MeshletGenPushConstants mgPush{
					.createTasksAddr = createAddr,
					.vertexPagePoolAddr = vertPoolAddr,
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
