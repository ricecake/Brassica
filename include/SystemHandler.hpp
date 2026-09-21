#pragma once

#include <memory>
#include <vector>

#include "passes/EntityNode.hpp"
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

		// Registers a new entity in the entt registry with at least basic spatial details (TransformComponent)
		entt::entity RegisterEntity(Engine& engine, const TransformComponent& transform = {});

		[[nodiscard]] const std::vector<entt::entity>& GetEntities() const { return m_entities; }

		[[nodiscard]] IEntityNode& GetEntityNode() {
			if (!m_entityNode) {
				m_entityNode = std::make_shared<EntityNode<SystemHandler>>();
			}
			return *m_entityNode;
		}

		[[nodiscard]] const IEntityNode& GetEntityNode() const {
			if (!m_entityNode) {
				m_entityNode = std::make_shared<EntityNode<SystemHandler>>();
			}
			return *m_entityNode;
		}

		virtual void InitNode(const render::NodeServices& services) {
			if (!m_nodeInitialized) {
				GetEntityNode().Init(services);
				m_nodeInitialized = true;
			}
		}

		virtual void DestroyNode(const vk::Device& device) {
			if (m_entityNode && m_nodeInitialized) {
				m_entityNode->Destroy(device);
				m_nodeInitialized = false;
			}
		}

		[[nodiscard]] bool IsNodeInitialized() const { return m_nodeInitialized; }

	protected:
		template <typename Tag>
		void CreateEntityNode() {
			m_entityNode = std::make_shared<EntityNode<Tag>>();
		}

		std::vector<entt::entity>            m_entities;
		mutable std::shared_ptr<IEntityNode> m_entityNode;
		bool                                 m_nodeInitialized{false};
	};

} // namespace brassica
