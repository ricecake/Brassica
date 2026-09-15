#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainAccelerationStructure.hpp"

namespace brassica {

	class ShaderWatcher;

	struct TerrainPushConstants {
		glm::uvec4 gridParams{8, 16, 2048, 1024}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::uvec4 lodOffsets0_3{0u};
		glm::uvec4 lodOffsets4_7{0u};
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t biomeIndex{0};
		std::uint32_t visibilityIndex{0};
		std::uint32_t indirectionMapIndex{0};
	};

	struct TerrainNode: render::NodeRegistrar<TerrainNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Create>,
			graph::Create<TerrainTLAS>,
			graph::Read<TerrainClipmapTexture>,
			graph::Read<TerrainMinMaxTexture>,
			graph::Read<TerrainBiomeTexture>,
			graph::Read<TerrainTileVisibilityTexture>,
			graph::Read<TerrainIndirectionMapTexture>>;

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
		TerrainAccelerationStructure* terrainAS = nullptr;
		TerrainPushConstants          push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			taskShader.CompileTaskFromFile(services.device, "shaders/terrain.task");
			meshShader.CompileMeshFromFile(services.device, "shaders/terrain.mesh");
			fragShader.CompileFragmentFromFile(services.device, "shaders/terrain.frag");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.gridParams = p.terrainGridParams;
			push.lodOffsets0_3 = p.terrainLodOffsets0_3;
			push.lodOffsets4_7 = p.terrainLodOffsets4_7;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainTLAS>(),
					.access = graph::AccessKind::Write,
					.desc = graph::AccelerationStructureDesc(),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.visibilityIndex = ctx.Index<TerrainTileVisibilityTexture>();
			push.indirectionMapIndex = ctx.Index<TerrainIndirectionMapTexture>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Unorm
			};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(TerrainPushConstants)
			}};
			render::GraphicsPipelineRequest      request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = vk::Format::eD32Sfloat,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, boundSets, nullptr);
			}

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(TerrainPushConstants),
				&push
			);

			uint32_t taskGroupCount = (push.gridParams.z + 31) / 32;
			if (terrainAS && terrainAS->GetDls().vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, terrainAS->GetDls());
			}
		}
	};

	BRASSICA_REGISTER_NODE(TerrainNode);

} // namespace brassica
