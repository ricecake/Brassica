#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	// Mirrors shaders/gltf_mesh.glsl's GltfVertex exactly, including the padding glm::vec3's
	// std430 struct-level alignment forces GLSL to insert but a plain C++ struct wouldn't add on
	// its own (glm::vec3's own alignof is 4, not 16). Offsets verified via spirv-dis against the
	// real compiled layout, not hand-computed -- see that file's own comment and
	// feedback_push_constant_struct_editing. PredefinedBufferNode sizes its upload as
	// data.size() * sizeof(T), so a mismatched sizeof() here would under-allocate the same way
	// the foliage archetype buffer did before that was caught.
	struct GltfVertex {
		glm::vec3 position;    // offset 0
		float     _pad0{0.0f}; // offset 12 -- pads to 16 for normal's vec3 alignment
		glm::vec3 normal;      // offset 16
		float     _pad1{0.0f}; // offset 28 -- pads to 32 for uv
		glm::vec2 uv;          // offset 32
		glm::vec2 _pad2{0.0f}; // offset 40 -- pads to 48 for joints' uvec4 alignment
		glm::uvec4 joints{0u}; // offset 48 -- JOINTS_0, zero-filled for non-skinned primitives
		glm::vec4  weights{0.0f}; // offset 64 -- WEIGHTS_0
	};

	static_assert(sizeof(GltfVertex) == 80, "GltfVertex must match gltf_mesh.glsl's verified 80-byte stride");

	// Mirrors shaders/gltf_mesh.glsl's GltfMeshlet exactly. Unlike GltfVertex, the real fields
	// here happen to pack tightly with no internal gaps (verified via spirv-dis) -- vec3 only
	// forces padding on whatever immediately follows if that follow-on member itself needs
	// stricter alignment, which none of these do (each vec3 here is followed by a plain scalar).
	// _pad0..2 only exist to round the trailing size up to the real 64-byte array stride.
	struct GltfMeshlet {
		std::uint32_t vertexOffset;
		std::uint32_t triangleOffset;
		std::uint32_t vertexCount;
		std::uint32_t triangleCount;
		glm::vec3     boundsCenter{0.0f};
		float         boundsRadius{0.0f};
		glm::vec3     coneAxis{0.0f};
		float         coneCutoff{0.0f};
		std::uint32_t materialIndex{0};
		std::uint32_t _pad0{0};
		std::uint32_t _pad1{0};
		std::uint32_t _pad2{0};
	};

	static_assert(sizeof(GltfMeshlet) == 64, "GltfMeshlet must match gltf_mesh.glsl's verified 64-byte stride");

} // namespace brassica
