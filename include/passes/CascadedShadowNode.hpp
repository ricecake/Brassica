#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	struct CascadedShadowBakePushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		};
		std::uint32_t shadowMapStorageIdx{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t activeCascadeMask{0xF};
	};

	struct CascadedShadowNode: render::NodeRegistrar<CascadedShadowNode> {
		using Resources = graph::Declares<
			graph::Create<CascadedShadowMapArray>,
			graph::Read<TerrainClipmapTexture>,
			graph::Read<TerrainMinMaxTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary*       pipelineLibrary = nullptr;
		ComputeShader                  compShader;
		CascadedShadowBakePushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/effects/cascaded_shadow_bake.comp")) {
				spdlog::critical("CascadedShadowNode compute shader compilation failed.");
				throw std::runtime_error("CascadedShadowNode compute shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.gridParams = p.terrainGridParams;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			graph::ResourceDesc desc = graph::ComputeStorageImageDesc(1024, 1024, vk::Format::eR32G32B32A32Sfloat);
			desc.layers = 4;
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CascadedShadowMapArray>(),
					.access = graph::AccessKind::Write,
					.desc = desc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.shadowMapStorageIdx = ctx.StorageIndex<CascadedShadowMapArray>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();

			std::uint64_t frameIdx = ctx.frameIndex;
			std::uint32_t mask = 0u;
			if (frameIdx % 1u == 0u) mask |= (1u << 0); // Cascade 0: every frame
			if (frameIdx % 2u == 0u) mask |= (1u << 1); // Cascade 1: every 2 frames
			if (frameIdx % 4u == 0u) mask |= (1u << 2); // Cascade 2: every 4 frames
			if (frameIdx % 8u == 0u) mask |= (1u << 3); // Cascade 3: every 8 frames
			push.activeCascadeMask = mask;

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CascadedShadowBakePushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(CascadedShadowBakePushConstants),
				&push
			);

			vkCmd.dispatch((1024 + 15) / 16, (1024 + 15) / 16, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CascadedShadowNode);

} // namespace brassica
