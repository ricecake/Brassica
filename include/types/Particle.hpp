#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	// Particle definition holding position, velocity, misc, type, lifetime, and maxLifetime.
	struct Particle {
		glm::vec4     position{0.0f, 0.0f, 0.0f, 1.0f};
		glm::vec4     velocity{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4     misc{0.0f}; // Miscellaneous purpose vector
		std::uint32_t type{0};
		float         lifetime{0.0f};
		float         maxLifetime{1.0f};
		std::uint32_t padding{0};
	};

	static_assert(sizeof(Particle) == 64, "Particle struct must be 64 bytes (16-byte aligned for GLSL std430)");

	// Particle type properties defining visual and physical characteristics for a particle type.
	struct ParticleType {
		glm::vec4 color{1.0f, 1.0f, 1.0f, 1.0f};
		float     size{0.1f};
		float     gravityScale{1.0f};
		float     drag{0.0f};
		float     padding{0.0f};
	};

	static_assert(sizeof(ParticleType) == 32, "ParticleType struct must be 32 bytes");

	// DrawMeshTasksIndirectCommandEXT structure matching Vulkan VK_EXT_mesh_shader layout.
	struct ParticleIndirectCommand {
		std::uint32_t groupCountX{0};
		std::uint32_t groupCountY{1};
		std::uint32_t groupCountZ{1};
	};

} // namespace brassica
