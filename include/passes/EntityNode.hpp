#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "VulkanCompat.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/IEntityNode.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

namespace brassica {

	template <typename Tag = struct DefaultEntityTag>
	struct EntityNode: public IEntityNode {
		using Resources = graph::Declares<GBuffer<graph::ModifyKey>, graph::Create<EntityIndirectBuffer<Tag>>>;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		const DispatchLoaderDynamic*    dls = nullptr;
		std::vector<EntityInstanceData> instances;
		MeshTasksIndirectCommand        indirectCmd{0, 0, 0};

		void AddInstance(const EntityPushConstants& push, const MeshTasksIndirectCommand& cmd = {1, 1, 1}) override {
			instances.push_back(EntityInstanceData{.push = push, .indirectCmd = cmd});
		}

		void ClearInstances() override {
			instances.clear();
		}

		void SetInstances(std::span<const EntityInstanceData> insts) override {
			instances.assign(insts.begin(), insts.end());
		}

		const std::vector<EntityInstanceData>& GetInstances() const override {
			return instances;
		}

		void SetPushConstants(const EntityPushConstants& p) override {
			if (instances.empty()) {
				instances.push_back(EntityInstanceData{.push = p, .indirectCmd = {1, 1, 1}});
			} else {
				instances[0].push = p;
			}
		}

		void SetIndirectCommand(const MeshTasksIndirectCommand& cmd) override {
			indirectCmd = cmd;
			if (instances.empty()) {
				instances.push_back(EntityInstanceData{.push = {}, .indirectCmd = cmd});
			} else {
				instances[0].indirectCmd = cmd;
			}
		}

		EntityPushConstants& GetPushConstants() override {
			if (instances.empty()) {
				instances.push_back(EntityInstanceData{});
			}
			return instances[0].push;
		}

		MeshTasksIndirectCommand& GetIndirectCommand() override {
			if (instances.empty()) {
				instances.push_back(EntityInstanceData{});
			}
			return instances[0].indirectCmd;
		}

		void RegisterInto(graph::Graph& graph) override { graph.RegisterRef(*this); }

		void Init(const render::NodeServices& services) override {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/ball.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/ball.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/ball.frag")) {
				spdlog::critical("EntityNode shader compilation failed.");
				throw std::runtime_error("EntityNode shader compilation failed.");
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

		void Destroy(vk::Device device) override {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.isActive = !instances.empty() || (indirectCmd.groupCountX > 0 || indirectCmd.groupCountY > 0 || indirectCmd.groupCountZ > 0);

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
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Srgb),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);

			graph::ResourceDesc indirectDesc = graph::MappedStorageBufferDesc(sizeof(MeshTasksIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<EntityIndirectBuffer<Tag>>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			ctx.WriteSpan<EntityIndirectBuffer<Tag>>(std::span<const MeshTasksIndirectCommand>(&indirectCmd, 1));

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
			std::array<vk::PushConstantRange, 1> pushConstantRanges{vk::PushConstantRange {
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(EntityPushConstants)
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

			vk::Buffer    indirectBuf{nullptr};
			std::uint64_t offset = 0;
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<EntityIndirectBuffer<Tag>>()) {
						indirectBuf = physBuf->GetBuffer();
						offset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
					}
				}
			}

			if (!instances.empty()) {
				for (const auto& inst : instances) {
					vkCmd.pushConstants(
						resolved.layout,
						vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
						0,
						sizeof(EntityPushConstants),
						&inst.push
					);

					if (dls && dls->vkCmdDrawMeshTasksEXT) {
						std::uint32_t groups = inst.indirectCmd.groupCountX > 0 ? inst.indirectCmd.groupCountX : 1;
						vkCmd.drawMeshTasksEXT(groups, 1, 1, *dls);
					}
				}
			} else if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					offset,
					1,
					sizeof(MeshTasksIndirectCommand)
				);
			} else if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t groups = indirectCmd.groupCountX > 0 ? indirectCmd.groupCountX : 1;
				vkCmd.drawMeshTasksEXT(groups, 1, 1, *dls);
			}
		}
	};

	using BallNode = EntityNode<struct BallSystemHandlerTag>;

} // namespace brassica
