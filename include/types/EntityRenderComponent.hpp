#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	struct EntityRenderComponent {
		std::uint32_t meshType{0}; // e.g. 0 = Sphere/Ball, 1 = Cylinder
		glm::vec4     color{0.0f, 0.4f, 1.0f, 1.0f};
		glm::uvec4    params{8, 12, 0, 0}; // Mesh mesh-specific rendering parameters
	};

	struct alignas(16) EntityInstanceData {
		glm::vec4  positionAndScale{0.0f, 0.0f, 0.0f, 1.0f}; // xyz = position, w = scale
		glm::vec4  color{1.0f};
		glm::uvec4 params{0};
	};

} // namespace brassica
