#pragma once

#include <vector>

#include <entt/entity/entity.hpp>

#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class Engine;

	class SystemHandler {
	public:
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

	protected:
		std::vector<entt::entity> m_entities;
	};

} // namespace brassica
