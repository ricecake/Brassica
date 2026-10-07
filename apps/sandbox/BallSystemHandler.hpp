#pragma once

#include <cmath>
#include <vector>

#include "Engine.hpp"
#include "SystemHandler.hpp"
#include "terrain/TerrainManager.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class BallSystemHandler: public SystemHandler {
	public:
		struct BallInstanceData {
			entt::entity entity{entt::null};
			float        spawnTime{0.0f};
			float        lifespan{5.0f};
			glm::vec3    basePos{0.0f};
			glm::vec3    velocity{0.0f};
		};

		BallSystemHandler() = default;

		void Setup(Engine& engine, const FrameDetails& frameDetails) override {
			m_spawnTimer = 0.0f;
			// Spawn initial batch of balls
			for (int i = 0; i < 5; ++i) {
				SpawnBall(engine, static_cast<float>(frameDetails.totalTime) - i * 0.8f);
			}
		}

		void Update(Engine& engine, const FrameDetails& frameDetails) override {
			float currentTime = static_cast<float>(frameDetails.totalTime);
			float dt = frameDetails.deltaTime > 0.0f ? frameDetails.deltaTime : 0.016f;

			m_spawnTimer += dt;
			if (m_spawnTimer >= m_spawnInterval && m_balls.size() < m_maxBalls) {
				m_spawnTimer = 0.0f;
				SpawnBall(engine, currentTime);
			}

			auto& registry = engine.GetRegistry();
			float terrainY = engine.GetTerrainManager().GetCachedGroundHeight(0.0f);

			for (auto it = m_balls.begin(); it != m_balls.end();) {
				float age = currentTime - it->spawnTime;
				if (age >= it->lifespan || !registry.valid(it->entity)) {
					UnregisterEntity(engine, it->entity);
					it = m_balls.erase(it);
					continue;
				}

				auto* transform = registry.try_get<TransformComponent>(it->entity);
				auto* renderComp = registry.try_get<EntityRenderComponent>(it->entity);

				if (transform && renderComp) {
					it->basePos += it->velocity * dt;
					transform->position = it->basePos;
					transform->position.y = terrainY + 10.0f + std::sin(age * 3.0f + it->basePos.x) * 2.0f;

					// Pulse color slightly over lifetime
					renderComp->color.a = std::sin(age / it->lifespan * 3.14159f); // Fade in & out
					renderComp->MarkDirty();
				}

				++it;
			}
		}

	private:
		void SpawnBall(Engine& engine, float currentTime) {
			float offset = static_cast<float>(m_balls.size()) * 3.0f - 6.0f;
			float terrainY = engine.GetTerrainManager().GetCachedGroundHeight(0.0f);

			TransformComponent transform{};
			transform.position = glm::vec3(offset, terrainY + 10.0f, -20.0f + (m_balls.size() % 3) * 2.0f);
			transform.scale = glm::vec3(2.5f + (m_balls.size() % 3) * 0.5f);

			EntityRenderComponent renderComp{};
			renderComp.meshType = EntityMeshType::Ball;
			renderComp.color = glm::vec4(
				0.1f + (m_balls.size() % 3) * 0.3f,
				0.4f + (m_balls.size() % 2) * 0.4f,
				1.0f - (m_balls.size() % 3) * 0.2f,
				1.0f
			);
			renderComp.meshParams = glm::uvec4(8, 12, 0, 0); // rings, pointsPerRing
			renderComp.material = glm::vec4(0.1f, 0.3f, 0.0f, 0.0f);

			entt::entity entity = RegisterEntity(engine, transform, renderComp);

			BallInstanceData ballData{};
			ballData.entity = entity;
			ballData.spawnTime = currentTime;
			ballData.lifespan = 4.0f + (m_balls.size() % 4) * 1.5f;
			ballData.basePos = transform.position;
			ballData.velocity = glm::vec3((m_balls.size() % 2 == 0 ? 1.0f : -1.0f) * 1.5f, 0.0f, (m_balls.size() % 3 - 1) * 1.0f);

			m_balls.push_back(ballData);
		}

		std::vector<BallInstanceData> m_balls;
		float                         m_spawnTimer{0.0f};
		float                         m_spawnInterval{1.2f};
		std::size_t                   m_maxBalls{12};
	};

} // namespace brassica
