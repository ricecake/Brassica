#include "SystemHandler.hpp"

#include <algorithm>

#include "Engine.hpp"
#include "passes/EntityNode.hpp"

namespace brassica {

	entt::entity SystemHandler::RegisterEntity(Engine& engine, const TransformComponent& transform, const EntityRenderComponent& render) {
		entt::entity entity = engine.GetRegistry().create();
		engine.GetRegistry().emplace<TransformComponent>(entity, transform);
		engine.GetRegistry().emplace<EntityRenderComponent>(entity, render);
		m_entities.push_back(entity);
		return entity;
	}

	bool SystemHandler::RemoveEntity(Engine& engine, entt::entity entity) {
		auto it = std::find(m_entities.begin(), m_entities.end(), entity);
		if (it != m_entities.end()) {
			m_entities.erase(it);
			if (engine.GetRegistry().valid(entity)) {
				engine.GetRegistry().destroy(entity);
			}
			return true;
		}
		return false;
	}

	void SystemHandler::ClearEntities(Engine& engine) {
		for (auto entity : m_entities) {
			if (engine.GetRegistry().valid(entity)) {
				engine.GetRegistry().destroy(entity);
			}
		}
		m_entities.clear();
	}

	IEntityNode& SystemHandler::GetEntityNode() {
		if (!m_entityNode) {
			m_entityNode = std::make_shared<EntityNode<SystemHandler>>();
		}
		return *m_entityNode;
	}

	const IEntityNode& SystemHandler::GetEntityNode() const {
		if (!m_entityNode) {
			m_entityNode = std::make_shared<EntityNode<SystemHandler>>();
		}
		return *m_entityNode;
	}

	void SystemHandler::InitNode(const render::NodeServices& services) {
		if (!m_nodeInitialized) {
			GetEntityNode().Init(services);
			m_nodeInitialized = true;
		}
	}

	void SystemHandler::DestroyNode(const vk::Device& device) {
		if (m_entityNode && m_nodeInitialized) {
			m_entityNode->Destroy(device);
			m_nodeInitialized = false;
		}
	}

} // namespace brassica
