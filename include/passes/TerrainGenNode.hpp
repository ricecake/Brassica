#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "terrain/TerrainAccelerationStructure.hpp"
#include "terrain/TerrainClipmap.hpp"

namespace brassica {

	struct TerrainGenPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		std::uint32_t clipmapStorageIdx{0};
		std::uint32_t minMaxStorageIdx{0};
		std::uint32_t biomeStorageIdx{0};
		std::uint32_t visibilityStorageIdx{0};
		bool forceRegeneration = true;
	};

	struct TerrainMinMaxMipPushConstants {
		glm::uvec4    gridParams{0};
		std::uint32_t srcMipIndex{0};
		std::uint32_t dstStorageIdx{0};
		std::uint32_t srcMip{0};
		std::uint32_t dstDim{0};
		bool          forceRegeneration = true;
	};

	struct TerrainHorizonPushConstants {
		glm::uvec4    gridParams{0};
		std::uint32_t clipmapIdx{0};
		std::uint32_t minMaxIdx{0};
		std::uint32_t horizonStorageIdx{0};
		bool          forceRegeneration = true;
	};

	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		using Resources = graph::Declares<
			graph::Modify<TerrainClipmapTexture>,
			graph::Modify<TerrainMinMaxTexture>,
			graph::Modify<TerrainMinMaxMip0>,
			graph::Modify<TerrainMinMaxMip1>,
			graph::Modify<TerrainMinMaxMip2>,
			graph::Modify<TerrainMinMaxMip3>,
			graph::Modify<TerrainMinMaxMip4>,
			graph::Modify<TerrainMinMaxMip5>,
			graph::Modify<TerrainMinMaxMip6>,
			graph::Modify<TerrainMinMaxMip7>,
			graph::Modify<TerrainMinMaxMip8>,
			graph::Modify<TerrainMinMaxMip9>,
			graph::Modify<TerrainMinMaxMip10>,
			graph::Modify<TerrainBiomeTexture>,
			graph::Modify<TerrainTileVisibilityTexture>,
			graph::Modify<TerrainHorizonTexture>,
			graph::Modify<TerrainTLAS>>;

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
		ComputeShader                    minmaxMipShader;
		ComputeShader                    horizonShader;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainGenPushConstants          push{};
		glm::vec3                        cameraPos{0.0f};
		bool                             hasUpdate{true};
		bool forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			physicalRegistry = services.physicalRegistry;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_gen.comp") ||
			    !aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp") ||
			    !minmaxMipShader.CompileComputeFromFile(services.device, "shaders/terrain_minmax_mip.comp") ||
			    !horizonShader.CompileComputeFromFile(services.device, "shaders/terrain_horizon.comp")) {
				spdlog::critical("TerrainGenNode shader compilation failed.");
				throw std::runtime_error("TerrainGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&aabbShader);
			watcher.RegisterShader(&minmaxMipShader);
			watcher.RegisterShader(&horizonShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
			minmaxMipShader.Destroy(device);
			horizonShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
			forceRegeneration = p.forceRegeneration;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(5);
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
					.key = graph::IdOf<TerrainMinMaxMip0>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(0, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip1>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(1, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip2>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(2, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip3>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(3, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip4>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(4, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip5>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(5, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip6>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(6, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip7>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(7, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip8>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(8, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip9>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(9, push.gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainMinMaxMip10>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainMinMaxMipDesc(10, push.gridParams.x),
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
					.key = graph::IdOf<TerrainHorizonTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainHorizonDesc(push.gridParams.x),
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

			if (hasUpdate || forceRegeneration) {
				push.clipmapStorageIdx = ctx.StorageIndex<TerrainClipmapTexture>();
				push.minMaxStorageIdx = ctx.StorageIndex<TerrainMinMaxTexture>();
				push.biomeStorageIdx = ctx.StorageIndex<TerrainBiomeTexture>();
				push.visibilityStorageIdx = ctx.StorageIndex<TerrainTileVisibilityTexture>();
				push.forceRegeneration = forceRegeneration;

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

				// Downsample min/max height mipmap levels 1..10
				uint32_t dstMipStorageIndices[11] = {
					ctx.StorageIndex<TerrainMinMaxMip0>(),
					ctx.StorageIndex<TerrainMinMaxMip1>(),
					ctx.StorageIndex<TerrainMinMaxMip2>(),
					ctx.StorageIndex<TerrainMinMaxMip3>(),
					ctx.StorageIndex<TerrainMinMaxMip4>(),
					ctx.StorageIndex<TerrainMinMaxMip5>(),
					ctx.StorageIndex<TerrainMinMaxMip6>(),
					ctx.StorageIndex<TerrainMinMaxMip7>(),
					ctx.StorageIndex<TerrainMinMaxMip8>(),
					ctx.StorageIndex<TerrainMinMaxMip9>(),
					ctx.StorageIndex<TerrainMinMaxMip10>()
				};

				std::array<vk::PushConstantRange, 1> mipPushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainMinMaxMipPushConstants)}
				};

				render::ComputePipelineRequest mipRequest{
					.shader = &minmaxMipShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = mipPushRanges,
				};
				render::ResolvedPipeline mipResolved = pipelineLibrary->ResolveCached(mipRequest);

				uint32_t totalMips = GetClipmapMipLevels(push.gridParams.w);
				for (uint32_t dstMip = 1; dstMip < totalMips && dstMip <= 10; ++dstMip) {
					vk::MemoryBarrier2 barrier(
						vk::PipelineStageFlagBits2::eComputeShader,
						vk::AccessFlagBits2::eShaderStorageWrite,
						vk::PipelineStageFlagBits2::eComputeShader,
						vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
					);
					vk::DependencyInfo dep({}, 1, &barrier);
					vkCmd.pipelineBarrier2(dep);

					uint32_t dstDim = std::max(1u, push.gridParams.w >> dstMip);
					TerrainMinMaxMipPushConstants mipPush{};
					mipPush.gridParams = push.gridParams;
					mipPush.srcMipIndex = ctx.Index<TerrainMinMaxTexture>();
					mipPush.dstStorageIdx = dstMipStorageIndices[dstMip];
					mipPush.srcMip = dstMip - 1;
					mipPush.dstDim = dstDim;
					mipPush.forceRegeneration = forceRegeneration;

					if (mipResolved.pipeline) {
						vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, mipResolved.pipeline);
						if (boundSets[0] && boundSets[1]) {
							vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, mipResolved.layout, 0, boundSets, nullptr);
						}
						vkCmd.pushConstants(
							mipResolved.layout,
							vk::ShaderStageFlagBits::eCompute,
							0,
							sizeof(TerrainMinMaxMipPushConstants),
							&mipPush
						);
						uint32_t mgX = (dstDim + 15) / 16;
						uint32_t mgY = (dstDim + 15) / 16;
						vkCmd.dispatch(mgX, mgY, push.gridParams.x);
					}
				}

				// Compute memory barrier before horizon map dispatch
				vk::MemoryBarrier2 horizonBarrier(
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
				);
				vk::DependencyInfo depHorizon({}, 1, &horizonBarrier);
				vkCmd.pipelineBarrier2(depHorizon);

				TerrainHorizonPushConstants horizonPush{};
				horizonPush.gridParams = push.gridParams;
				horizonPush.clipmapIdx = ctx.Index<TerrainClipmapTexture>();
				horizonPush.minMaxIdx = ctx.Index<TerrainMinMaxTexture>();
				horizonPush.horizonStorageIdx = ctx.StorageIndex<TerrainHorizonTexture>();
				horizonPush.forceRegeneration = forceRegeneration;

				std::array<vk::PushConstantRange, 1> horizonPushRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainHorizonPushConstants)}
				};

				render::ComputePipelineRequest horizonRequest{
					.shader = &horizonShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = horizonPushRanges,
				};
				render::ResolvedPipeline horizonResolved = pipelineLibrary->ResolveCached(horizonRequest);

				if (horizonResolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, horizonResolved.pipeline);
					if (boundSets[0] && boundSets[1]) {
						vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, horizonResolved.layout, 0, boundSets, nullptr);
					}
					vkCmd.pushConstants(
						horizonResolved.layout,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sizeof(TerrainHorizonPushConstants),
						&horizonPush
					);
					vkCmd.dispatch(groupX, groupY, groupZ);
				}
			}

			if (false && terrainAS) {
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
					push.gridParams
				);
				if (physicalRegistry && terrainAS->GetTLAS()) {
					physicalRegistry->RegisterImportedAccelerationStructure<TerrainTLAS>(terrainAS->GetTLAS());
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainGenNode);

} // namespace brassica
