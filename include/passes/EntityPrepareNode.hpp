#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "VulkanCompat.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/TransformComponent.hpp"
#include <entt/entity/registry.hpp>

namespace brassica {

	struct EntityPrepareNode : render::NodeRegistrar<EntityPrepareNode> {
		using Resources = graph::Declares<
			graph::Create<EntityInstanceBuffer>,
			graph::Create<EntityIndirectBuffer>
		>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		// Transient execution storage
		std::vector<EntityInstanceData> m_batchedInstances;
		MeshTasksIndirectCommand        m_indirectCommand{0, 0, 0};
		std::uint64_t                   m_bufferDeviceAddress{0};
		entt::registry*                 m_registry{nullptr};

		void Init(const render::NodeServices& /*services*/) {}
		void Destroy(vk::Device /*device*/) {}

		void SetFrameParams(const render::NodeFrameParams& params) {
			m_registry = params.registry;
		}

		graph::Recipe Setup(const graph::FrameContext& /*ctx*/) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};

			// Capacity for up to 1024 entity instances
			std::size_t instBufferSize = sizeof(EntityInstanceData) * 1024;
			if (instBufferSize == 0) instBufferSize = 1024;

			graph::ResourceDesc instDesc = graph::MappedStorageBufferDesc(instBufferSize);
			instDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eShaderDeviceAddress);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<EntityInstanceBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = instDesc,
				}
			);

			graph::ResourceDesc indirectDesc = graph::MappedStorageBufferDesc(sizeof(MeshTasksIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<EntityIndirectBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			m_batchedInstances.clear();

			// 1. Collect entities from EnTT registry if available
			if (m_registry) {
				CollectAndBatchEntities(*m_registry);
			}

			// 2. Upload instances to host-mapped EntityInstanceBuffer
			if (!m_batchedInstances.empty()) {
				ctx.WriteSpan<EntityInstanceBuffer>(
					std::span<const EntityInstanceData>(m_batchedInstances.data(), m_batchedInstances.size())
				);
			}

			// 3. Populate indirect command
			std::uint32_t totalCount = static_cast<std::uint32_t>(m_batchedInstances.size());
			m_indirectCommand.groupCountX = totalCount > 0 ? totalCount : 0;
			m_indirectCommand.groupCountY = 1;
			m_indirectCommand.groupCountZ = 1;
			ctx.WriteSpan<EntityIndirectBuffer>(
				std::span<const MeshTasksIndirectCommand>(&m_indirectCommand, 1)
			);

			// 4. Query Buffer Device Address for EntityInstanceBuffer
			m_bufferDeviceAddress = 0;
			if (ctx.resources) {
				if (const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = physReg->GetBuffer<EntityInstanceBuffer>()) {
						vk::Buffer vkBuf = physBuf->GetBuffer();
						if (vkBuf) {
							vk::BufferDeviceAddressInfo bdaInfo(vkBuf);
							m_bufferDeviceAddress = physReg->GetDevice().getBufferAddress(bdaInfo);
							std::uint64_t sliceOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
							m_bufferDeviceAddress += sliceOffset;
						}
					}
				}
			}
		}

		void CollectAndBatchEntities(entt::registry& registry) {
			m_batchedInstances.clear();
			auto view = registry.view<TransformComponent, EntityRenderComponent>();

			struct Entry {
				entt::entity entity;
				EntityInstanceData inst;
			};
			std::vector<Entry> entries;
			entries.reserve(view.size_hint());

			for (auto entity : view) {
				const auto& transform = view.get<TransformComponent>(entity);
				auto& renderComp = view.get<EntityRenderComponent>(entity);

				EntityInstanceData inst{};
				inst.positionAndScale = glm::vec4(transform.position, transform.scale.x);
				inst.color = renderComp.color;
				inst.params = glm::uvec4(
					static_cast<std::uint32_t>(renderComp.meshType),
					renderComp.meshParams.x,
					renderComp.meshParams.y,
					renderComp.meshParams.z
				);
				inst.material = renderComp.material;
				inst.rotation = glm::vec4(transform.rotation, 0.0f);

				Entry entry{};
				entry.entity = entity;
				entry.inst = inst;
				entries.push_back(entry);

				renderComp.gpuVersion = renderComp.version;
				renderComp.isDirty = false;
			}

			// Sort by mesh type to batch draw calls
			std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
				return a.inst.params.x < b.inst.params.x;
			});

			for (const auto& entry : entries) {
				m_batchedInstances.push_back(entry.inst);
			}
		}

		[[nodiscard]] std::uint64_t GetBufferDeviceAddress() const { return m_bufferDeviceAddress; }
		[[nodiscard]] std::uint32_t GetTotalInstances() const { return static_cast<std::uint32_t>(m_batchedInstances.size()); }
	};

	BRASSICA_REGISTER_NODE(EntityPrepareNode);

} // namespace brassica
