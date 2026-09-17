#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct BallPushConstants {
		glm::vec4 positionAndScale{0.0f, 15.0f, 0.0f, 3.0f};
		glm::vec4 color{0.0f, 0.4f, 1.0f, 1.0f}; // Bright blue
		glm::uvec4 params{8, 12, 0, 0};          // rings, pointsPerRing
	};

	struct MeshTasksIndirectCommand {
		std::uint32_t groupCountX{1};
		std::uint32_t groupCountY{1};
		std::uint32_t groupCountZ{1};
	};

	struct BallNode: render::NodeRegistrar<BallNode> {
		using Resources = graph::Declares<GBuffer<graph::Modify>, graph::Read<BallIndirectBuffer>>;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		TaskShader                   taskShader;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		const DispatchLoaderDynamic* dls = nullptr;
		BallPushConstants            push{};

		static BallPushConstants        s_currentPush;
		static MeshTasksIndirectCommand s_indirectCmd;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			taskShader.CompileTaskFromFile(services.device, "shaders/ball.task");
			meshShader.CompileMeshFromFile(services.device, "shaders/ball.mesh");
			fragShader.CompileFragmentFromFile(services.device, "shaders/ball.frag");
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

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(MeshTasksIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<BallIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push = s_currentPush;

			ctx.WriteSpan<BallIndirectBuffer>(
				std::span<const MeshTasksIndirectCommand>(&s_indirectCmd, 1)
			);

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
				vk::Format::eR32G32B32A32Sfloat,
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
				sizeof(BallPushConstants)
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
				sizeof(BallPushConstants),
				&push
			);

			vk::Buffer indirectBuf{nullptr};
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<BallIndirectBuffer>()) {
						indirectBuf = physBuf->GetBuffer();
					}
				}
			}

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					0,
					1,
					sizeof(MeshTasksIndirectCommand)
				);
			} else if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t groups = s_indirectCmd.groupCountX > 0 ? s_indirectCmd.groupCountX : 1;
				vkCmd.drawMeshTasksEXT(groups, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(BallNode);

} // namespace brassica
