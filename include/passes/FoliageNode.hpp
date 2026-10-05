#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "foliage/IFoliageManager.hpp"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	struct FoliagePushConstants {
		glm::uvec4 gridParams{8u, 16u, 2048u, 1088u}; // x = numLODs, y = tilesPerRow, z = totalTiles, w = textureDim
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t biomeIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		float         windTime{0.0f};
		float         lodBaseRange{20.0f};
		float         lodScaleFactor{2.0f};
		float         baseTileSize{16.0f};
	};

	struct FoliageNode: render::NodeRegistrar<FoliageNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Modify>,
			graph::Read<TerrainClipmapTexture>,
			graph::Read<TerrainMinMaxTexture>,
			graph::Read<TerrainBiomeTexture>,
			graph::Read<TerrainWeatherBiomeTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::GBuffer;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eGreaterOrEqual,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		TaskShader                   taskShader;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		const DispatchLoaderDynamic* dls = nullptr;
		FoliagePushConstants         push{};
		bool                         enabled{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/foliage.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/foliage.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/foliage.frag")) {
				spdlog::critical("FoliageNode shader compilation failed.");
				throw std::runtime_error("FoliageNode shader compilation failed.");
			}
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
			push.windTime = p.time;
			if (ServiceLocator::Instance().Has<IFoliageManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IFoliageManager>();
				enabled = mgr->IsEnabled();
				auto props = mgr->GetGlobalProperties();
				push.lodBaseRange = props.lodBaseRange;
				push.lodScaleFactor = props.lodScaleFactor;
			}
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Graphics,
				.isActive = enabled,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferPosition>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferNormal>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferAlbedo>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Srgb),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferDepth>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
				vk::Format::eR32G32B32A32Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Srgb
			};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(FoliagePushConstants)
			}};

			render::GraphicsPipelineRequest request{
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
				sizeof(FoliagePushConstants),
				&push
			);

			uint32_t taskGroupCount = (push.gridParams.z + 31) / 32;
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(FoliageNode);

} // namespace brassica
