#include "SystemHandler.hpp"

#include "Engine.hpp"
#include <algorithm>

namespace brassica {

	entt::entity SystemHandler::RegisterEntity(
		Engine& engine,
		const TransformComponent& transform,
		const EntityRenderComponent& renderComp
	) {
		entt::entity entity = engine.GetRegistry().create();
		engine.GetRegistry().emplace<TransformComponent>(entity, transform);
		engine.GetRegistry().emplace<EntityRenderComponent>(entity, renderComp);
		m_entities.push_back(entity);
		return entity;
	}

	void SystemHandler::UnregisterEntity(Engine& engine, entt::entity entity) {
		auto it = std::find(m_entities.begin(), m_entities.end(), entity);
		if (it != m_entities.end()) {
			m_entities.erase(it);
		}
		if (engine.GetRegistry().valid(entity)) {
			engine.GetRegistry().destroy(entity);
		}
	}

	void SystemHandler::ClearEntities(Engine& engine) {
		for (auto entity : m_entities) {
			if (engine.GetRegistry().valid(entity)) {
				engine.GetRegistry().destroy(entity);
			}
		}
		m_entities.clear();
	}

	void SystemHandler::MarkEntityDirty(Engine& engine, entt::entity entity) {
		if (engine.GetRegistry().valid(entity)) {
			auto* renderComp = engine.GetRegistry().try_get<EntityRenderComponent>(entity);
			if (renderComp) {
				renderComp->MarkDirty();
			}
		}
	}

} // namespace brassica
