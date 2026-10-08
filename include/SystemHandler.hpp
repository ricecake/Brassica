#pragma once

#include <memory>
#include <vector>

#include "VulkanCompat.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"
#include <entt/entity/entity.hpp>

namespace brassica {

	class Engine;

	namespace render {
		struct NodeServices;
	}

	class SystemHandler {
	public:
		SystemHandler() = default;
		virtual ~SystemHandler() = default;

		// Required setup method
		virtual void Setup(Engine& engine, const FrameDetails& frameDetails) = 0;

		// Optional callbacks
		virtual void PreFrame(Engine& /*engine*/, const FrameDetails& /*frameDetails*/) {}

		virtual void PostFrame(Engine& /*engine*/, const FrameDetails& /*frameDetails*/) {}

		virtual void Cleanup(Engine& /*engine*/) {}

		// Method passed an entity for updates
		virtual void UpdateEntity(entt::entity /*entity*/, Engine& /*engine*/, const FrameDetails& /*frameDetails*/) {}

		// Default frame update iterating entities managed by this handler
		virtual void Update(Engine& engine, const FrameDetails& frameDetails) {
			auto it = m_entities.begin();
			while (it != m_entities.end()) {
				auto entity = *it;
				UpdateEntity(entity, engine, frameDetails);
				// If UpdateEntity destroyed the entity, check valid or let handler manage m_entities
				++it;
			}
		}

		// Registers a new entity in the entt registry with TransformComponent and EntityRenderComponent
		entt::entity RegisterEntity(
			Engine& engine,
			const TransformComponent& transform = {},
			const EntityRenderComponent& renderComp = {}
		);

		// Unregisters and destroys an entity from the registry and m_entities list
		void UnregisterEntity(Engine& engine, entt::entity entity);

		// Clears all entities managed by this SystemHandler from registry
		void ClearEntities(Engine& engine);

		// Marks an entity dirty to signal render buffer update
		void MarkEntityDirty(Engine& engine, entt::entity entity);

		[[nodiscard]] const std::vector<entt::entity>& GetEntities() const { return m_entities; }

		virtual void InitNode(const render::NodeServices& /*services*/) {}
		virtual void DestroyNode(const vk::Device& /*device*/) {}

	protected:
		std::vector<entt::entity> m_entities;
	};

} // namespace brassica
