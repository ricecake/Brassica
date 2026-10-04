#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "VulkanCompat.hpp"
#include <glm/glm.hpp>

namespace graph {
	class Graph;
}

namespace brassica {

	namespace render {
		struct NodeServices;
	}

	struct EntityPushConstants {
		glm::vec4  positionAndScale{0.0f, 15.0f, 0.0f, 3.0f};
		glm::vec4  color{0.0f, 0.4f, 1.0f, 1.0f}; // Bright blue
		glm::uvec4 params{8, 12, 0, 0};           // rings/verts, pointsPerRing/indices, materialId, flags
	};

	using BallPushConstants = EntityPushConstants;

	struct MeshTasksIndirectCommand {
		std::uint32_t groupCountX{1};
		std::uint32_t groupCountY{1};
		std::uint32_t groupCountZ{1};
	};

	struct EntityRenderComponent {
		std::uint32_t entityTypeId{0};
		glm::vec4     color{0.0f, 0.4f, 1.0f, 1.0f};
		std::uint32_t materialId{0};
		glm::uvec4    params{8, 12, 0, 0};
	};

	struct EntityInstanceData {
		EntityPushConstants      push{};
		MeshTasksIndirectCommand indirectCmd{1, 1, 1};
	};

	struct IEntityNode {
		virtual ~IEntityNode() = default;
		virtual void                      Init(const render::NodeServices& services) = 0;
		virtual void                      Destroy(vk::Device device) = 0;
		virtual void                      RegisterInto(graph::Graph& graph) = 0;
		virtual void                      SetPushConstants(const EntityPushConstants& p) = 0;
		virtual void                      SetIndirectCommand(const MeshTasksIndirectCommand& cmd) = 0;
		virtual EntityPushConstants&      GetPushConstants() = 0;
		virtual MeshTasksIndirectCommand& GetIndirectCommand() = 0;

		// Instance batching interface
		virtual void                                   AddInstance(const EntityPushConstants& push, const MeshTasksIndirectCommand& cmd = {1, 1, 1}) = 0;
		virtual void                                   ClearInstances() = 0;
		virtual void                                   SetInstances(std::span<const EntityInstanceData> instances) = 0;
		virtual const std::vector<EntityInstanceData>& GetInstances() const = 0;
	};

} // namespace brassica
