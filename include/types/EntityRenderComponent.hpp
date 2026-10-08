#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	enum class EntityMeshType : std::uint32_t {
		Ball = 0,
		Cylinder = 1,
		Cow = 2
	};

	struct EntityInstanceData {
		glm::vec4     positionAndScale{0.0f, 0.0f, 0.0f, 1.0f};
		glm::vec4     color{1.0f, 1.0f, 1.0f, 1.0f};
		glm::uvec4    params{0, 0, 0, 0};   // x = meshTypeId, y = param1 (rings/vertexCount), z = param2 (pointsPerRing/triangleCount), w = flags
		glm::vec4     material{0.0f, 0.5f, 0.0f, 0.0f}; // x = metallic, y = roughness, z = glint, w = emissivity
		glm::vec4     rotation{0.0f, 0.0f, 0.0f, 0.0f}; // xyz = euler angles (radians)
		std::uint64_t vertexBufferAddress{0};
		std::uint64_t indexBufferAddress{0};
	};

	struct EntityRenderComponent {
		EntityMeshType meshType{EntityMeshType::Ball};
		glm::vec4      color{0.0f, 0.4f, 1.0f, 1.0f};
		glm::uvec4     meshParams{8, 12, 0, 0}; // x = rings/vertexCount, y = pointsPerRing/triangleCount
		glm::vec4      material{0.0f, 0.5f, 0.0f, 0.0f}; // metallic, roughness, glint, emissivity
		std::uint64_t  vertexBufferAddress{0};
		std::uint64_t  indexBufferAddress{0};

		std::uint32_t  version{1};
		std::uint32_t  gpuVersion{0};
		bool           isDirty{true};

		void MarkDirty() {
			version++;
			isDirty = true;
		}
	};

	struct EntityPushConstants {
		std::uint64_t entityBufferAddress{0};
		std::uint32_t totalInstances{0};
		std::uint32_t baseInstanceIndex{0};
	};

	struct MeshTasksIndirectCommand {
		std::uint32_t groupCountX{1};
		std::uint32_t groupCountY{1};
		std::uint32_t groupCountZ{1};
	};

} // namespace brassica
