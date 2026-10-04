#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "passes/IEntityNode.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"
#include <entt/entity/entity.hpp>

namespace vk {
	class Device;
}

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
			for (auto entity : m_entities) {
				UpdateEntity(entity, engine, frameDetails);
			}
		}

		// Registers a new entity in the entt registry with spatial and optional render details
		entt::entity RegisterEntity(Engine& engine, const TransformComponent& transform = {}, const EntityRenderComponent& render = {});

		// Removes an entity managed by this handler from m_entities and destroys it in the entt registry
		bool RemoveEntity(Engine& engine, entt::entity entity);

		// Clears all entities managed by this handler
		void ClearEntities(Engine& engine);

		[[nodiscard]] const std::vector<entt::entity>& GetEntities() const { return m_entities; }

		[[nodiscard]] IEntityNode& GetEntityNode();

		[[nodiscard]] const IEntityNode& GetEntityNode() const;

		virtual void InitNode(const render::NodeServices& services);

		virtual void DestroyNode(const vk::Device& device);

		[[nodiscard]] bool IsNodeInitialized() const { return m_nodeInitialized; }

	protected:
		template <typename NodeT, typename... Args>
		void CreateEntityNode(Args&&... args) {
			m_entityNode = std::make_shared<NodeT>(std::forward<Args>(args)...);
		}

		std::vector<entt::entity>            m_entities;
		mutable std::shared_ptr<IEntityNode> m_entityNode;
		bool                                 m_nodeInitialized{false};
	};

} // namespace brassica
