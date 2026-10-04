#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "VulkanCompat.hpp"

#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	struct EntityPushConstants {
		std::uint64_t instanceBufferAddress{0}; // 64-bit device address
		std::uint32_t baseInstanceIndex{0};     // base offset into the instance buffer for this batch
		std::uint32_t totalInstances{0};        // total instances in this batch
		std::uint32_t flags{0};                 // general rendering flags / params
	};

	using BallPushConstants = EntityPushConstants;

	struct MeshTasksIndirectCommand {
		std::uint32_t groupCountX{1};
		std::uint32_t groupCountY{1};
		std::uint32_t groupCountZ{1};
	};

	struct IEntityNode {
		virtual ~IEntityNode() = default;
		virtual void                      Init(const render::NodeServices& services) = 0;
		virtual void                      Destroy(vk::Device device) = 0;
		virtual void                      RegisterInto(graph::Graph& graph) = 0;
		virtual void                      SetPushConstants(const EntityPushConstants& p) = 0;
		virtual void                      SetIndirectCommand(const MeshTasksIndirectCommand& cmd) = 0;
		virtual EntityPushConstants&      GetPushConstants() = 0;
		virtual MeshTasksIndirectCommand& GetIndirectCommand() = 0;

		virtual void AddInstance(const TransformComponent& transform, const EntityRenderComponent& renderComp = {}) = 0;
		virtual void ClearInstances() = 0;
	};

	template <typename Tag = struct DefaultEntityTag>
	struct EntityNode: public IEntityNode {
		using Resources = graph::Declares<
			GBuffer<graph::ModifyKey>,
			graph::Create<EntityInstanceBuffer<Tag>>,
			graph::Create<EntityIndirectBuffer<Tag>>
		>;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		struct InternalInstance {
			TransformComponent    transform;
			EntityRenderComponent render;
		};

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		const DispatchLoaderDynamic* dls = nullptr;
		EntityPushConstants          push{};
		MeshTasksIndirectCommand     indirectCmd{0, 0, 0};

		std::vector<InternalInstance> m_rawInstances;

		void SetPushConstants(const EntityPushConstants& p) override { push = p; }

		void SetIndirectCommand(const MeshTasksIndirectCommand& cmd) override { indirectCmd = cmd; }

		EntityPushConstants& GetPushConstants() override { return push; }

		MeshTasksIndirectCommand& GetIndirectCommand() override { return indirectCmd; }

		void AddInstance(const TransformComponent& transform, const EntityRenderComponent& renderComp = {}) override {
			m_rawInstances.push_back({transform, renderComp});
		}

		void ClearInstances() override {
			m_rawInstances.clear();
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
			r.isActive = true;

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

			std::size_t maxInstances = std::max<std::size_t>(m_rawInstances.size(), 1);
			graph::ResourceDesc instanceDesc = graph::MappedStorageBufferDesc(sizeof(EntityInstanceData) * maxInstances);
			instanceDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<EntityInstanceBuffer<Tag>>(),
					.access = graph::AccessKind::Write,
					.desc = instanceDesc,
				}
			);

			std::size_t maxBatches = std::max<std::size_t>(m_rawInstances.size(), 1);
			graph::ResourceDesc indirectDesc = graph::MappedStorageBufferDesc(sizeof(MeshTasksIndirectCommand) * maxBatches);
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
			// Sort local instances by meshType to reduce state switches and batch efficiently
			std::vector<InternalInstance> sortedInstances = m_rawInstances;
			std::sort(sortedInstances.begin(), sortedInstances.end(), [](const InternalInstance& a, const InternalInstance& b) {
				return a.render.meshType < b.render.meshType;
			});

			std::vector<EntityInstanceData>       gpuInstances;
			std::vector<MeshTasksIndirectCommand> indirectCommands;

			gpuInstances.reserve(sortedInstances.size());

			for (const auto& inst : sortedInstances) {
				EntityInstanceData gpuInst{};
				gpuInst.positionAndScale = glm::vec4(inst.transform.position, inst.transform.scale.x);
				gpuInst.color = inst.render.color;
				gpuInst.params = inst.render.params;

				gpuInstances.push_back(gpuInst);
			}

			if (!gpuInstances.empty()) {
				indirectCommands.push_back(MeshTasksIndirectCommand{static_cast<std::uint32_t>(gpuInstances.size()), 1, 1});
			} else if (indirectCmd.groupCountX > 0 || indirectCmd.groupCountY > 0 || indirectCmd.groupCountZ > 0) {
				indirectCommands.push_back(indirectCmd);
			}

			if (!gpuInstances.empty()) {
				ctx.WriteSpan<EntityInstanceBuffer<Tag>>(std::span<const EntityInstanceData>(gpuInstances.data(), gpuInstances.size()));
			}
			if (!indirectCommands.empty()) {
				ctx.WriteSpan<EntityIndirectBuffer<Tag>>(std::span<const MeshTasksIndirectCommand>(indirectCommands.data(), indirectCommands.size()));
			}

			vk::Buffer    indirectBuf{nullptr};
			std::uint64_t indirectOffset = 0;
			std::uint64_t instanceDeviceAddress = 0;

			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physIndirect = registry->GetBuffer<EntityIndirectBuffer<Tag>>()) {
						indirectBuf = physIndirect->GetBuffer();
						indirectOffset = physIndirect->SliceStride() * (ctx.frameIndex % physIndirect->RingSlots());
					}
					if (auto physInst = registry->GetBuffer<EntityInstanceBuffer<Tag>>()) {
						vk::BufferDeviceAddressInfo addrInfo{physInst->GetBuffer()};
						instanceDeviceAddress = registry->GetDevice().getBufferAddress(addrInfo);
						instanceDeviceAddress += physInst->SliceStride() * (ctx.frameIndex % physInst->RingSlots());
					}
				}
			}

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

			push.instanceBufferAddress = instanceDeviceAddress;
			push.baseInstanceIndex = 0;
			push.totalInstances = static_cast<std::uint32_t>(gpuInstances.size());

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(EntityPushConstants),
				&push
			);

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf && !indirectCommands.empty()) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					indirectOffset,
					static_cast<std::uint32_t>(indirectCommands.size()),
					sizeof(MeshTasksIndirectCommand)
				);
			} else if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t groups = !gpuInstances.empty() ? static_cast<std::uint32_t>(gpuInstances.size()) : (indirectCmd.groupCountX > 0 ? indirectCmd.groupCountX : 1);
				vkCmd.drawMeshTasksEXT(groups, 1, 1, *dls);
			}
		}
	};

	using BallNode = EntityNode<struct BallSystemHandlerTag>;

} // namespace brassica
