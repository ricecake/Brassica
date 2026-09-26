#pragma once

#include <cmath>

#include "Engine.hpp"
#include "SystemHandler.hpp"
#include "animation/SkinnedCylinder.hpp"
#include "passes/EntityNode.hpp"
#include "terrain/TerrainClipmap.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class OzzCylinderSystemHandler: public SystemHandler {
	public:
		OzzCylinderSystemHandler() { CreateEntityNode<OzzCylinderSystemHandler>(); }

		void Setup(Engine& engine, const FrameDetails& frameDetails) override {
			float targetX = 6.0f;
			float targetZ = -20.0f;
			float terrainY = TerrainClipmap::SampleTerrain(targetX, targetZ, 0.5f).r;

			TransformComponent transform{};
			transform.position = glm::vec3(targetX, terrainY + 8.0f, targetZ);
			transform.rotation = glm::vec3(0.0f);
			transform.scale = glm::vec3(3.0f);

			cylinderEntity = RegisterEntity(engine, transform);

			cylinderColor = glm::vec4(0.9f, 0.4f, 0.1f, 1.0f); // Bright orange

			MeshTasksIndirectCommand cmd{};
			cmd.groupCountX = 1;
			cmd.groupCountY = 1;
			cmd.groupCountZ = 1;
			GetEntityNode().SetIndirectCommand(cmd);

			UpdateRenderData(transform, frameDetails.totalTime);
		}

		void UpdateEntity(entt::entity entity, Engine& engine, const FrameDetails& frameDetails) override {
			if (entity != cylinderEntity) {
				return;
			}

			auto& registry = engine.GetRegistry();
			if (!registry.valid(entity)) {
				return;
			}

			m_cylinder.Update(static_cast<float>(frameDetails.totalTime));

			auto* transform = registry.try_get<TransformComponent>(entity);
			if (transform) {
				float targetX = 6.0f;
				float targetZ = -20.0f;
				float terrainY = TerrainClipmap::SampleTerrain(targetX, targetZ, 0.5f).r;

				transform->position.x = targetX;
				transform->position.z = targetZ;
				transform->position.y = terrainY + 8.0f;
				UpdateRenderData(*transform, frameDetails.totalTime);
			}
		}

		[[nodiscard]] const SkinnedCylinder& GetSkinnedCylinder() const { return m_cylinder; }

	private:
		void UpdateRenderData(const TransformComponent& transform, double totalTime) {
			EntityPushConstants push{};
			push.positionAndScale = glm::vec4(transform.position, transform.scale.x);
			push.color = cylinderColor;
			push.params = glm::uvec4(
				static_cast<uint32_t>(m_cylinder.GetVertexCount()),
				static_cast<uint32_t>(m_cylinder.GetTriangleCount()),
				0,
				0
			);

			GetEntityNode().SetPushConstants(push);
		}

		entt::entity    cylinderEntity{entt::null};
		glm::vec4       cylinderColor{0.9f, 0.4f, 0.1f, 1.0f};
		SkinnedCylinder m_cylinder;
	};

} // namespace brassica
