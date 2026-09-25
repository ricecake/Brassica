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

	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		using Resources = graph::Declares<
			graph::Modify<TerrainClipmapTexture>,
			graph::Modify<TerrainMinMaxTexture>,
			graph::Modify<TerrainBiomeTexture>,
			graph::Modify<TerrainTileVisibilityTexture>,
			graph::Modify<TerrainTLAS>>;

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
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
			    !aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp")) {
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
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
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
				forceRegeneration = false;
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
