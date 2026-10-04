#include "SystemHandler.hpp"

#include <algorithm>
#include "Engine.hpp"

namespace brassica {

	void SystemHandler::SyncRenderInstances(Engine& engine) {
		if (&engine == nullptr || !m_entityNode) return;
		m_entityNode->ClearInstances();
		auto& registry = engine.GetRegistry();
		for (auto entity : m_entities) {
			if (!registry.valid(entity)) continue;
			const auto* transform = registry.try_get<TransformComponent>(entity);
			const auto* render = registry.try_get<EntityRenderComponent>(entity);
			if (transform) {
				EntityRenderComponent renderComp = render ? *render : EntityRenderComponent{};
				m_entityNode->AddInstance(*transform, renderComp);
			}
		}
	}

	entt::entity SystemHandler::RegisterEntity(Engine& engine, const TransformComponent& transform, const EntityRenderComponent& render) {
		entt::entity entity = engine.GetRegistry().create();
		engine.GetRegistry().emplace<TransformComponent>(entity, transform);
		engine.GetRegistry().emplace<EntityRenderComponent>(entity, render);
		m_entities.push_back(entity);
		SyncRenderInstances(engine);
		return entity;
	}

	void SystemHandler::RemoveEntity(Engine& engine, entt::entity entity) {
		auto it = std::find(m_entities.begin(), m_entities.end(), entity);
		if (it != m_entities.end()) {
			m_entities.erase(it);
		}
		if (engine.GetRegistry().valid(entity)) {
			engine.GetRegistry().destroy(entity);
		}
		SyncRenderInstances(engine);
	}

	void SystemHandler::ClearEntities(Engine& engine) {
		for (auto entity : m_entities) {
			if (engine.GetRegistry().valid(entity)) {
				engine.GetRegistry().destroy(entity);
			}
		}
		m_entities.clear();
		if (m_entityNode) {
			m_entityNode->ClearInstances();
		}
	}

} // namespace brassica
