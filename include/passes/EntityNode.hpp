#pragma once

#include <array>
#include <cstdint>

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

namespace brassica {

	struct EntityPushConstants {
		glm::vec4  positionAndScale{0.0f, 15.0f, 0.0f, 3.0f};
		glm::vec4  color{0.0f, 0.4f, 1.0f, 1.0f}; // Bright blue
		glm::uvec4 params{8, 12, 0, 0};           // rings, pointsPerRing
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
	};

	template <typename Tag = struct DefaultEntityTag>
	struct EntityNode: public IEntityNode {
		// GBuffer<graph::Modify>, not Create: TerrainNode is the real Create<GBuffer...> producer
		// (GBuffer<graph::Create> there), and this node's own Setup() below realizes all four
		// GBuffer keys with AccessKind::ReadWrite (a depth-tested opaque draw into the *existing*
		// gbuffer, not a fresh one) -- declaring Create here claimed to be a second, independent
		// producer of the same keys with no dependency on Terrain's write, and DeferredNode's
		// Read<GBuffer> had no reason to end up ordered after this node's draw either. Modify
		// consumes Terrain's Create output and re-produces it, so Deferred's Read now picks up
		// this node's write too -- the same last-writer-wins chaining WaterNode/
		// AtmosphereCompositeNode already rely on for Modify<HdrColor>.
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

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		const DispatchLoaderDynamic* dls = nullptr;
		EntityPushConstants          push{};
		MeshTasksIndirectCommand     indirectCmd{0, 0, 0};

		void SetPushConstants(const EntityPushConstants& p) override { push = p; }

		void SetIndirectCommand(const MeshTasksIndirectCommand& cmd) override { indirectCmd = cmd; }

		EntityPushConstants& GetPushConstants() override { return push; }

		MeshTasksIndirectCommand& GetIndirectCommand() override { return indirectCmd; }

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
			r.isActive = (indirectCmd.groupCountX > 0 || indirectCmd.groupCountY > 0 || indirectCmd.groupCountZ > 0);

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
				vk::Format::eR8G8B8A8Unorm
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

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(EntityPushConstants),
				&push
			);

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

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
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
