#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "Engine.hpp"
#include "SystemHandler.hpp"
#include "animation/OzzModel.hpp"
#include "terrain/TerrainManager.hpp"
#include "types/EntityRenderComponent.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class CowSystemHandler : public SystemHandler {
	public:
		struct CowInstanceData {
			entt::entity      entity{entt::null};
			float             spawnTime{0.0f};
			glm::vec3         basePos{0.0f};
			glm::vec3         velocity{0.0f};
			std::size_t       animIndex{0};
			float             motionChangeTimer{0.0f};
			float             motionDuration{4.0f};
			OzzModelInstance  modelInstance;
		};

		CowSystemHandler() = default;

		void Setup(Engine& engine, const FrameDetails& frameDetails) override {
			m_cowModel = std::make_unique<OzzModel>("assets/cow.glb");

			m_cows.clear();
			std::size_t cowCount = 6;
			float currentTime = static_cast<float>(frameDetails.totalTime);

			for (std::size_t i = 0; i < cowCount; ++i) {
				SpawnCow(engine, currentTime, i);
			}
		}

		void Update(Engine& engine, const FrameDetails& frameDetails) override {
			if (!m_cowModel || m_cowModel->GetAnimationCount() == 0) return;

			float currentTime = static_cast<float>(frameDetails.totalTime);
			float dt = frameDetails.deltaTime > 0.0f ? frameDetails.deltaTime : 0.016f;

			auto& registry = engine.GetRegistry();

			for (auto& cow : m_cows) {
				if (!registry.valid(cow.entity)) continue;

				cow.motionChangeTimer += dt;
				if (cow.motionChangeTimer >= cow.motionDuration) {
					cow.motionChangeTimer = 0.0f;
					cow.animIndex = (cow.animIndex + 1) % m_cowModel->GetAnimationCount();
					cow.motionDuration = 3.0f + static_cast<float>(rand() % 4);
				}

				// Update unique OzzModelInstance animation state for this individual cow
				m_cowModel->UpdateInstance(cow.modelInstance, dt, cow.animIndex);

				auto* transform = registry.try_get<TransformComponent>(cow.entity);
				auto* renderComp = registry.try_get<EntityRenderComponent>(cow.entity);

				if (transform && renderComp) {
					cow.basePos += cow.velocity * dt;

					// Bound position around center
					if (std::abs(cow.basePos.x) > 40.0f) cow.velocity.x *= -1.0f;
					if (std::abs(cow.basePos.z) > 40.0f) cow.velocity.z *= -1.0f;

					float groundY = engine.GetTerrainManager().GetCachedGroundHeight(cow.basePos.x);

					transform->position = cow.basePos;
					transform->position.y = groundY + 0.5f;

					if (glm::length(cow.velocity) > 0.01f) {
						transform->rotation.y = std::atan2(cow.velocity.x, cow.velocity.z);
					}

					float playbackTime = cow.modelInstance.playbackTime;
					renderComp->meshParams = glm::uvec4(
						static_cast<uint32_t>(m_cowModel->GetVertexCount()),
						static_cast<uint32_t>(m_cowModel->GetTriangleCount()),
						static_cast<uint32_t>(cow.animIndex),
						static_cast<uint32_t>(playbackTime * 1000.0f)
					);
					renderComp->material.w = playbackTime;
					renderComp->MarkDirty();
				}
			}
		}

		[[nodiscard]] const OzzModel* GetCowModel() const { return m_cowModel.get(); }

	private:
		void SpawnCow(Engine& engine, float currentTime, std::size_t index) {
			float angle = static_cast<float>(index) * (2.0f * 3.14159f / 6.0f);
			float radius = 15.0f + static_cast<float>(index % 3) * 5.0f;

			glm::vec3 pos(std::cos(angle) * radius, 0.0f, std::sin(angle) * radius - 15.0f);
			float groundY = engine.GetTerrainManager().GetCachedGroundHeight(pos.x);
			pos.y = groundY + 0.5f;

			TransformComponent transform{};
			transform.position = pos;
			transform.scale = glm::vec3(1.5f);

			EntityRenderComponent renderComp{};
			renderComp.meshType = EntityMeshType::Cow;
			renderComp.color = glm::vec4(0.92f, 0.88f, 0.8f, 1.0f);
			renderComp.meshParams = glm::uvec4(
				static_cast<uint32_t>(m_cowModel->GetVertexCount()),
				static_cast<uint32_t>(m_cowModel->GetTriangleCount()),
				static_cast<uint32_t>(index % m_cowModel->GetAnimationCount()),
				0
			);
			renderComp.material = glm::vec4(0.05f, 0.5f, 0.0f, 0.0f);

			entt::entity entity = RegisterEntity(engine, transform, renderComp);

			CowInstanceData cowData{};
			cowData.entity = entity;
			cowData.spawnTime = currentTime;
			cowData.basePos = transform.position;
			cowData.velocity = glm::vec3(
				std::sin(angle * 2.0f) * 1.5f,
				0.0f,
				std::cos(angle * 2.0f) * 1.5f
			);
			cowData.animIndex = index % m_cowModel->GetAnimationCount();
			cowData.motionChangeTimer = 0.0f;
			cowData.motionDuration = 4.0f + static_cast<float>(index % 3);
			cowData.modelInstance = m_cowModel->CreateInstance();

			m_cows.push_back(cowData);
		}

		std::unique_ptr<OzzModel>    m_cowModel;
		std::vector<CowInstanceData> m_cows;
	};

} // namespace brassica
