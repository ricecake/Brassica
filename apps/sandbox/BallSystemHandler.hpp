#pragma once

#include <cmath>

#include "Engine.hpp"
#include "SystemHandler.hpp"
#include "passes/BallNode.hpp"
#include "types/FrameDetails.hpp"
#include "types/TransformComponent.hpp"

namespace brassica {

	class BallSystemHandler: public SystemHandler {
	public:
		void Setup(Engine& engine, const FrameDetails& frameDetails) override {
			TransformComponent transform{};
			transform.position = glm::vec3(0.0f, 15.0f, 0.0f);
			transform.rotation = glm::vec3(0.0f);
			transform.scale = glm::vec3(3.0f);

			ballEntity = RegisterEntity(engine, transform);

			ballColor = glm::vec4(0.0f, 0.4f, 1.0f, 1.0f); // Bright blue
			rings = 8;
			pointsPerRing = 12;

			MeshTasksIndirectCommand cmd{};
			cmd.groupCountX = 1;
			cmd.groupCountY = 1;
			cmd.groupCountZ = 1;
			BallNode::s_indirectCmd = cmd;

			UpdateRenderData(transform);
		}

		void UpdateEntity(entt::entity entity, Engine& engine, const FrameDetails& frameDetails) override {
			if (entity != ballEntity) {
				return;
			}

			auto& registry = engine.GetRegistry();
			if (!registry.valid(entity)) {
				return;
			}

			auto* transform = registry.try_get<TransformComponent>(entity);
			if (transform) {
				transform->position.y = 15.0f + static_cast<float>(std::sin(frameDetails.totalTime * 2.0)) * 2.0f;
				UpdateRenderData(*transform);
			}
		}

	private:
		void UpdateRenderData(const TransformComponent& transform) {
			BallPushConstants push{};
			push.positionAndScale = glm::vec4(transform.position, transform.scale.x);
			push.color = ballColor;
			push.params = glm::uvec4(rings, pointsPerRing, 0, 0);

			BallNode::s_currentPush = push;
		}

		entt::entity ballEntity{entt::null};
		glm::vec4    ballColor{0.0f, 0.4f, 1.0f, 1.0f};
		uint32_t     rings{8};
		uint32_t     pointsPerRing{12};
	};

} // namespace brassica
