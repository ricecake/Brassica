#include "SystemHandler.hpp"
#include "Engine.hpp"

namespace brassica {

	entt::entity SystemHandler::RegisterEntity(Engine& engine, const TransformComponent& transform) {
		entt::entity entity = engine.GetRegistry().create();
		engine.GetRegistry().emplace<TransformComponent>(entity, transform);
		m_entities.push_back(entity);
		return entity;
	}

} // namespace brassica
