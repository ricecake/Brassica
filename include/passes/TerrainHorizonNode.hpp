#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "terrain/TerrainManager.hpp"

namespace brassica {

	struct TerrainHorizonPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		std::uint32_t clipmapStorageIdx{0};
		std::uint32_t horizonStorageIdx{0};
		std::uint32_t mode{0}; // 0 = horizontal rows, 1 = vertical columns
		std::uint32_t forceRegeneration{1};
	};

	struct TerrainHorizonNode: render::NodeRegistrar<TerrainHorizonNode> {
		using Resources = graph::Declares<
			graph::Read<TerrainClipmapTexture>,
			graph::Create<TerrainHorizonTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            horizonShader;
		TerrainHorizonPushConstants push{};
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		};
		bool hasUpdate{true};
		bool forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!horizonShader.CompileComputeFromFile(services.device, "shaders/terrain_horizon.comp")) {
				spdlog::critical("TerrainHorizonNode shader compilation failed.");
				throw std::runtime_error("TerrainHorizonNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&horizonShader);
			}
		}

		void Destroy(vk::Device device) {
			horizonShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
			forceRegeneration = p.forceRegeneration;
		}

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainClipmapTexture>(),
					.access = graph::AccessKind::Read,
					.desc = TerrainClipmapDesc(gridParams.x),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainHorizonTexture>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = TerrainHorizonDesc(gridParams.x),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (!hasUpdate && !forceRegeneration) {
				return;
			}

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout))
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet))
			};

			push.gridParams = gridParams;
			push.clipmapStorageIdx = ctx.Index<TerrainClipmapTexture>();
			push.horizonStorageIdx = ctx.StorageIndex<TerrainHorizonTexture>();
			push.forceRegeneration = forceRegeneration ? 1u : 0u;

			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainHorizonPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &horizonShader,
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

			// Mode 0: horizontal rows
			push.mode = 0u;
			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainHorizonPushConstants), &push);
			vkCmd.dispatch(1, gridParams.w, gridParams.x);

			// Memory barrier between mode 0 and mode 1
			vk::MemoryBarrier2 memBarrier{
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageWrite,
				vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite
			};
			vk::DependencyInfo depInfo{};
			depInfo.setMemoryBarriers(memBarrier);
			vkCmd.pipelineBarrier2(depInfo);

			// Mode 1: vertical columns
			push.mode = 1u;
			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TerrainHorizonPushConstants), &push);
			vkCmd.dispatch(1, gridParams.w, gridParams.x);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainHorizonNode);

} // namespace brassica
