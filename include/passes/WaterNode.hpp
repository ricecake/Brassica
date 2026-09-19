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
#include "VulkanCompat.hpp"

namespace brassica {

	class ShaderWatcher;

	struct WaterPushConstants {
		glm::uvec4    gridParams{14, 16, 3584, 1088}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::vec3     waterColor{0.05f, 0.45f, 0.85f};
		float         waterLevel{0.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t biomeIndex{0};
		std::uint32_t visibilityIndex{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t padding0{0};
		std::uint32_t padding1{0};
		glm::uvec4    lodOffsets0_3{0u};
		glm::uvec4    lodOffsets4_7{0u};
	};

	struct WaterNode: render::NodeRegistrar<WaterNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Read>,
			graph::Read<TerrainBiomeTexture>,
			graph::Read<TerrainTileVisibilityTexture>,
			graph::Read<TerrainClipmapTexture>,
			graph::Modify<Swapchain>
		>;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		// enableShadingRate=false is a real design choice, not just test convenience: this
		// fragment shader is a couple of texture reads and arithmetic, not the kind of expensive
		// per-pixel work VRS exists to coarsen (Terrain's LOD shading, Deferred's ray-query
		// shadows) -- and running it at full resolution avoids any risk of blocky artifacts at a
		// translucent blend boundary. It's also what makes WaterNode the one real, visible node
		// in this migration that's fully testable on a device without mesh-shader/ray-query/VRS
		// support (tests/test_water_node.cpp) rather than only compiling.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
			.depthCompareOp = vk::CompareOp::eLessOrEqual,
			.enableBlend = true,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		TaskShader                   taskShader;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		const DispatchLoaderDynamic* dls = nullptr;
		vk::Format                   swapchainFormat = vk::Format::eUndefined;
		WaterPushConstants           push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			swapchainFormat = services.swapchainFormat;
			taskShader.CompileTaskFromFile(services.device, "shaders/water.task");
			meshShader.CompileMeshFromFile(services.device, "shaders/water.mesh");
			fragShader.CompileFragmentFromFile(services.device, "shaders/water.frag");
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
			push.gridParams = glm::uvec4(14, 16, 3584, 1088);
			push.waterColor = p.waterColor;
			push.waterLevel = p.waterLevel;
			push.lodOffsets0_3 = p.terrainLodOffsets0_3;
			push.lodOffsets4_7 = p.terrainLodOffsets4_7;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Read,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.visibilityIndex = ctx.Index<TerrainTileVisibilityTexture>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();

			std::array<GraphicsShader*, 3>         stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
					vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(WaterPushConstants)
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
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
					vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(WaterPushConstants),
				&push
			);

			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t taskGroupCount = (push.gridParams.z + 31) / 32;
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(WaterNode);

} // namespace brassica
