#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	/// Sentinel value used to represent unbound or unused texture slots.
	constexpr std::uint32_t UNBOUND_TEXTURE_ID = 0xFFFFFFFFu;

	/// Procedural shading role a material plays. Drives which noise function (if any)
	/// gltf.frag evaluates for fragments using this material -- replaces unconditionally
	/// applying leaf noise to every model regardless of what it actually is. Standard must stay
	/// 0 so every already-registered material (glTF imports, bark/rock/flower/etc.) keeps
	/// rendering through the plain PBR path unless explicitly opted into a procedural role.
	enum class MaterialRole : std::uint32_t {
		Standard = 0,
		ProceduralLeaf = 1,
		ProceduralBark = 2,
		ProceduralRock = 3,
		ProceduralNeedle = 4,
		ProceduralCactus = 5,
	};

	/**
	 * @brief CPU/GPU material representation with std430 memory layout.
	 *
	 * - textureIds: uvec4 for albedo (x), normal (y), occlusion/roughness/metallic (z), emission (w)
	 *   texture IDs with sentinel value UNBOUND_TEXTURE_ID for unbound or unused slots.
	 * - colorTintOffset: vec4 for color tint offset (x), roughness strength (y),
	 *   metallic strength (z), and emission strength (w).
	 * - materialRole: which procedural noise role (if any) this material plays; see MaterialRole.
	 * - noiseScale/noiseWarp/noiseSharpness: generic knobs the role's noise function interprets
	 *   in its own domain (e.g. world-space frequency, domain-warp strength, cell/edge crispness) --
	 *   deliberately generic rather than one field per role, so leaf/bark/rock noise can share
	 *   the same three dials instead of each needing its own set.
	 * - uvScaleOffset: vec4 for mapping or displacement (x,y: scale; z,w: offset).
	 * - palette: four vec3s for a cosine color palette (stored as 4 x vec4 for 16-byte std430 alignment).
	 * - baseColorFactor/emissiveFactor: appended, not interleaved (see
	 *   feedback_push_constant_struct_editing) -- real RGBA color factors, for a photoreal
	 *   textured/flat-colored material (e.g. a glTF import). colorTintOffset's x/w scalars are a
	 *   different, procedural-palette-oriented contract (see evaluateCosinePalette,
	 *   material_system.glsl) that can't represent an arbitrary base or emissive color; this was a
	 *   real gap, not something to approximate by reusing those scalars.
	 * - proceduralShading: x is the blend strength between the macro (geometric) normal and the
	 *   procedural role's own normal (0 = macro only, 1 = fully procedural); yzw reserved for
	 *   role-specific tuning not generic enough for noiseScale/Warp/Sharpness (e.g. leaf blade
	 *   anisotropy, bark ridge direction bias).
	 */
	struct alignas(16) MaterialData {
		/// Texture indices (x: albedo, y: normal, z: occlusion/roughness/metallic, w: emission)
		glm::uvec4 textureIds{UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID, UNBOUND_TEXTURE_ID};

		/// Color tint offset (x), roughness strength (y), metallic strength (z), emission strength (w)
		glm::vec4 colorTintOffset{0.0f, 1.0f, 0.0f, 0.0f};

		/// Procedural shading role (see MaterialRole); stored raw so this struct can be memcpy'd
		/// straight to the GPU like every other field here.
		std::uint32_t materialRole{static_cast<std::uint32_t>(MaterialRole::Standard)};
		/// World-space noise frequency multiplier for the role's noise function
		float noiseScale{1.0f};
		/// Domain-warp strength applied to the noise domain before the role's noise function evaluates
		float noiseWarp{0.0f};
		/// Cell/edge crispness produced by the role's noise function
		float noiseSharpness{0.5f};

		/// UV scale (xy) and offset (zw)
		glm::vec4 uvScaleOffset{1.0f, 1.0f, 0.0f, 0.0f};

		/// Four vec3s for cosine color palette: a + b * cos(2*PI*(c*t + d))
		glm::vec4 palette[4]{
			glm::vec4(0.5f, 0.5f, 0.5f, 0.0f),
			glm::vec4(0.5f, 0.5f, 0.5f, 0.0f),
			glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),
			glm::vec4(0.0f, 0.33f, 0.67f, 0.0f)
		};

		/// RGBA base-color multiply (applied on top of the sampled albedo texture, or the flat
		/// color itself when textureIds.x is unbound). w is unused (alpha blending isn't wired up
		/// yet) but kept for std430 alignment and glTF baseColorFactor's own RGBA shape.
		glm::vec4 baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};

		/// RGB emissive multiply (applied on top of the sampled emissive texture, or the flat
		/// emissive color itself when textureIds.w is unbound); colorTintOffset.w ("emission
		/// strength") further scales this. w is unused, kept for alignment.
		glm::vec4 emissiveFactor{0.0f, 0.0f, 0.0f, 0.0f};

		/// x: normal blend strength for the procedural role (0 = macro normal, 1 = fully
		/// procedural); yzw reserved for role-specific tuning. See MaterialRole doc above.
		glm::vec4 proceduralShading{0.0f, 0.0f, 0.0f, 0.0f};
	};

	static_assert(sizeof(MaterialData) == 176, "MaterialData struct size must be 176 bytes");

} // namespace brassica
