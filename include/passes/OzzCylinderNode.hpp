#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "VulkanCompat.hpp"

#include "EngineConstants.hpp"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/EntityNode.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

namespace brassica {

	struct CylinderGpuVertex {
		glm::vec4 position{0.0f, 0.0f, 0.0f, 1.0f};
		glm::vec4 normal{0.0f, 1.0f, 0.0f, 0.0f};
	};

	template <typename Tag = struct OzzCylinderTag>
	struct OzzCylinderNode: public IEntityNode {
		// Plain unversioned GBuffer<graph::ModifyKey>, same as EntityNode's: both this node and
		// EntityNode run at the same implicit Phase::Default, and Graph::CollectEdges now resolves
		// same-phase self-modifiers of the same key into an automatic registration-order chain
		// rather than requiring either node to hand-pick an explicit version number (see
		// CollectEdges's own comment, Graph.hpp). Neither entity type needs to know the other
		// exists.
		using Resources = graph::Declares<
			GBuffer<graph::ModifyKey>,
			graph::Create<CylinderVertexBuffer>,
			graph::Create<CylinderIndexBuffer>,
			graph::Create<EntityIndirectBuffer<Tag>>
		>;

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
		MeshTasksIndirectCommand     indirectCmd{1, 1, 1};

		std::vector<CylinderGpuVertex> meshVertices;
		std::vector<std::uint32_t>     meshIndices;

		vk::DescriptorSetLayout set2Layout;
		vk::DescriptorPool      set2Pool;
		std::array<vk::DescriptorSet, FRAME_OVERLAP> set2Sets{};

		void SetPushConstants(const EntityPushConstants& p) override { push = p; }
		void SetIndirectCommand(const MeshTasksIndirectCommand& cmd) override { indirectCmd = cmd; }
		EntityPushConstants& GetPushConstants() override { return push; }
		MeshTasksIndirectCommand& GetIndirectCommand() override { return indirectCmd; }

		void AddInstance(const TransformComponent& /*transform*/, const EntityRenderComponent& /*renderComp*/ = {}) override {}
		void ClearInstances() override {}

		void SetMeshData(const std::vector<glm::vec3>& positions, const std::vector<glm::vec3>& normals, const std::vector<std::uint32_t>& indices) {
			meshVertices.resize(positions.size());
			for (size_t i = 0; i < positions.size(); ++i) {
				meshVertices[i].position = glm::vec4(positions[i], 1.0f);
				meshVertices[i].normal = glm::vec4(normals[i], 0.0f);
			}
			meshIndices = indices;
		}

		void RegisterInto(graph::Graph& graph) override { graph.RegisterRef(*this); }

		void Init(const render::NodeServices& services) override {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/cylinder.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/cylinder.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cylinder.frag")) {
				spdlog::critical("OzzCylinderNode shader compilation failed.");
				throw std::runtime_error("OzzCylinderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			// Create Set 2 Layout with 2 storage buffers
			std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
				vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT},
				vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT}
			};
			vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
			set2Layout = services.device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 2 * FRAME_OVERLAP}
			};
			vk::DescriptorPoolCreateInfo poolInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, FRAME_OVERLAP, poolSizes);
			set2Pool = services.device.createDescriptorPool(poolInfo);

			std::array<vk::DescriptorSetLayout, FRAME_OVERLAP> layouts{set2Layout, set2Layout};
			vk::DescriptorSetAllocateInfo allocInfo(set2Pool, FRAME_OVERLAP, layouts.data());
			auto sets = services.device.allocateDescriptorSets(allocInfo);
			for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
				set2Sets[i] = sets[i];
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) override {
			if (set2Pool) {
				device.destroyDescriptorPool(set2Pool);
				set2Pool = nullptr;
			}
			if (set2Layout) {
				device.destroyDescriptorSetLayout(set2Layout);
				set2Layout = nullptr;
			}
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
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Srgb),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferMaterial>(),
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

			graph::ResourceDesc vertDesc = graph::MappedStorageBufferDesc(sizeof(CylinderGpuVertex) * 512);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CylinderVertexBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = vertDesc,
				}
			);

			graph::ResourceDesc indexDesc = graph::MappedStorageBufferDesc(sizeof(std::uint32_t) * 2048);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CylinderIndexBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = indexDesc,
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
			if (!meshVertices.empty()) {
				ctx.WriteSpan<CylinderVertexBuffer>(std::span<const CylinderGpuVertex>(meshVertices.data(), meshVertices.size()));
			}
			if (!meshIndices.empty()) {
				ctx.WriteSpan<CylinderIndexBuffer>(std::span<const std::uint32_t>(meshIndices.data(), meshIndices.size()));
			}
			ctx.WriteSpan<EntityIndirectBuffer<Tag>>(std::span<const MeshTasksIndirectCommand>(&indirectCmd, 1));

			uint32_t activeFrame = ctx.frameIndex % FRAME_OVERLAP;
			vk::DescriptorSet activeSet = set2Sets[activeFrame];

			// Retrieve physical buffers and update Set 2 descriptors
			vk::Buffer vertBuf{nullptr};
			std::uint64_t vertOffset = 0;
			vk::Buffer indexBuf{nullptr};
			std::uint64_t indexOffset = 0;
			vk::Buffer indirectBuf{nullptr};
			std::uint64_t indirectOffset = 0;

			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<CylinderVertexBuffer>()) {
						vertBuf = physBuf->GetBuffer();
						vertOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
					}
					if (auto physBuf = registry->GetBuffer<CylinderIndexBuffer>()) {
						indexBuf = physBuf->GetBuffer();
						indexOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
					}
					if (auto physBuf = registry->GetBuffer<EntityIndirectBuffer<Tag>>()) {
						indirectBuf = physBuf->GetBuffer();
						indirectOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
					}

					if (vertBuf && indexBuf && activeSet && !meshVertices.empty() && !meshIndices.empty()) {
						vk::DescriptorBufferInfo vertBufferInfo(vertBuf, vertOffset, sizeof(CylinderGpuVertex) * meshVertices.size());
						vk::DescriptorBufferInfo indexBufferInfo(indexBuf, indexOffset, sizeof(std::uint32_t) * meshIndices.size());

						std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
							vk::WriteDescriptorSet(activeSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &vertBufferInfo),
							vk::WriteDescriptorSet(activeSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &indexBufferInfo)
						};
						registry->GetDevice().updateDescriptorSets(descriptorWrites, nullptr);
					}
				}
			}

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
				vk::Format::eR32G32B32A32Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Srgb,
				vk::Format::eR8G8B8A8Unorm
			};
			std::array<vk::DescriptorSetLayout, 3> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				set2Layout
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

			std::array<vk::DescriptorSet, 3> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet),
				activeSet
			};
			if (boundSets[0] && boundSets[1] && boundSets[2]) {
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

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					indirectOffset,
					1,
					sizeof(MeshTasksIndirectCommand)
				);
			} else if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t groups = indirectCmd.groupCountX > 0 ? indirectCmd.groupCountX : 1;
				vkCmd.drawMeshTasksEXT(groups, 1, 1, *dls);
			}
		}
	};

} // namespace brassica
