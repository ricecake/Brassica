#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace brassica {

	/// Sentinel value used to represent unbound or unused texture slots.
	constexpr std::uint32_t UNBOUND_TEXTURE_ID = 0xFFFFFFFFu;

	/**
	 * @brief CPU/GPU material representation with std430 memory layout.
	 *
	 * - textureIds: uvec4 for albedo (x), normal (y), occlusion/roughness/metallic (z), emission (w)
	 *   texture IDs with sentinel value UNBOUND_TEXTURE_ID for unbound or unused slots.
	 * - colorTintOffset: vec4 for color tint offset (x), roughness strength (y),
	 *   metallic strength (z), and emission strength (w).
	 * - shadingBitmask: uint shading control flow bitmask.
	 * - uvScaleOffset: vec4 for mapping or displacement (x,y: scale; z,w: offset).
	 * - palette: four vec3s for a cosine color palette (stored as 4 x vec4 for 16-byte std430 alignment).
	 */
	struct alignas(16) MaterialData {
		/// Texture indices (x: albedo, y: normal, z: occlusion/roughness/metallic, w: emission)
		glm::uvec4 textureIds{UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID};

		/// Color tint offset (x), roughness strength (y), metallic strength (z), emission strength (w)
		glm::vec4 colorTintOffset{0.0f, 1.0f, 0.0f, 0.0f};

		/// Shading control flow bitmask
		std::uint32_t shadingBitmask{0};
		std::uint32_t padding[3]{0, 0, 0};

		/// UV scale (xy) and offset (zw)
		glm::vec4 uvScaleOffset{1.0f, 1.0f, 0.0f, 0.0f};

		/// Four vec3s for cosine color palette: a + b * cos(2*PI*(c*t + d))
		glm::vec4 palette[4]{
			glm::vec4(0.5f, 0.5f, 0.5f, 0.0f),
			glm::vec4(0.5f, 0.5f, 0.5f, 0.0f),
			glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
			glm::vec4(0.0f, 0.33f, 0.67f, 0.0f)
		};
	};

	static_assert(sizeof(MaterialData) == 128, "MaterialData struct size must be 128 bytes");

} // namespace brassica
