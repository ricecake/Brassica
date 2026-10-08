#pragma once

#include <cmath>
#include <vector>

#include "animation/SkinnedCylinder.hpp"
#include "Engine.hpp"
#include "SystemHandler.hpp"
#include "terrain/TerrainManager.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class OzzCylinderSystemHandler: public SystemHandler {
	public:
		struct CylinderInstanceData {
			entt::entity entity{entt::null};
			float        spawnTime{0.0f};
			float        lifespan{6.0f};
			glm::vec3    basePos{0.0f};
			glm::vec3    velocity{0.0f};
		};

		OzzCylinderSystemHandler() = default;

		void Setup(Engine& engine, const FrameDetails& frameDetails) override {
			m_spawnTimer = 0.0f;
			for (int i = 0; i < 4; ++i) {
				SpawnCylinder(engine, static_cast<float>(frameDetails.totalTime) - i * 1.0f);
			}
		}

		void Update(Engine& engine, const FrameDetails& frameDetails) override {
			float currentTime = static_cast<float>(frameDetails.totalTime);
			float dt = frameDetails.deltaTime > 0.0f ? frameDetails.deltaTime : 0.016f;

			m_cylinder.Update(currentTime);

			m_spawnTimer += dt;
			if (m_spawnTimer >= m_spawnInterval && m_cylinders.size() < m_maxCylinders) {
				m_spawnTimer = 0.0f;
				SpawnCylinder(engine, currentTime);
			}

			auto& registry = engine.GetRegistry();
			float terrainY = engine.GetTerrainManager().GetCachedGroundHeight(0.0f);

			for (auto it = m_cylinders.begin(); it != m_cylinders.end();) {
				float age = currentTime - it->spawnTime;
				if (age >= it->lifespan || !registry.valid(it->entity)) {
					UnregisterEntity(engine, it->entity);
					it = m_cylinders.erase(it);
					continue;
				}

				auto* transform = registry.try_get<TransformComponent>(it->entity);
				auto* renderComp = registry.try_get<EntityRenderComponent>(it->entity);

				if (transform && renderComp) {
					it->basePos += it->velocity * dt;
					transform->position = it->basePos;
					transform->position.y = terrainY + 6.0f + std::cos(age * 2.0f + it->basePos.x) * 1.5f;
					transform->rotation.y = age * 0.5f;

					renderComp->color.a = std::sin(age / it->lifespan * 3.14159f);
					renderComp->MarkDirty();
				}

				++it;
			}
		}

		[[nodiscard]] const SkinnedCylinder& GetSkinnedCylinder() const { return m_cylinder; }

	private:
		void SpawnCylinder(Engine& engine, float currentTime) {
			float offset = static_cast<float>(m_cylinders.size()) * 4.0f - 6.0f;
			float terrainY = engine.GetTerrainManager().GetCachedGroundHeight(0.0f);

			TransformComponent transform{};
			transform.position = glm::vec3(offset, terrainY + 6.0f, -25.0f + (m_cylinders.size() % 2) * 4.0f);
			transform.scale = glm::vec3(2.5f);

			EntityRenderComponent renderComp{};
			renderComp.meshType = EntityMeshType::Cylinder;
			renderComp.color = glm::vec4(
				0.9f - (m_cylinders.size() % 3) * 0.2f,
				0.4f + (m_cylinders.size() % 2) * 0.3f,
				0.1f + (m_cylinders.size() % 3) * 0.2f,
				1.0f
			);
			renderComp.meshParams = glm::uvec4(
				static_cast<uint32_t>(m_cylinder.GetVertexCount()),
				static_cast<uint32_t>(m_cylinder.GetTriangleCount()),
				0,
				0
			);
			renderComp.material = glm::vec4(0.2f, 0.4f, 0.0f, 0.0f);

			entt::entity entity = RegisterEntity(engine, transform, renderComp);

			CylinderInstanceData cylData{};
			cylData.entity = entity;
			cylData.spawnTime = currentTime;
			cylData.lifespan = 5.0f + (m_cylinders.size() % 3) * 1.5f;
			cylData.basePos = transform.position;
			cylData.velocity = glm::vec3((m_cylinders.size() % 2 == 0 ? -1.0f : 1.0f) * 1.2f, 0.0f, 0.5f);

			m_cylinders.push_back(cylData);
		}

		std::vector<CylinderInstanceData> m_cylinders;
		SkinnedCylinder                    m_cylinder;
		float                              m_spawnTimer{0.0f};
		float                              m_spawnInterval{1.5f};
		std::size_t                        m_maxCylinders{8};
	};

} // namespace brassica
