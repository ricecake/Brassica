#ifndef HELPERS_MATERIAL_SYSTEM_GLSL
#define HELPERS_MATERIAL_SYSTEM_GLSL

#include "bindless.glsl"

const uint UNBOUND_TEXTURE_ID = 0xFFFFFFFFu;

/**
 * @brief GLSL MaterialData matching std430 C++ MaterialData struct layout.
 */
struct MaterialData {
	uvec4 texture_ids;         // x: albedo, y: normal, z: occlusion/roughness/metallic, w: emission
	vec4  color_tint_offset;   // x: color tint offset, y: roughness, z: metallic, w: emission strength
	uint  material_role;       // procedural shading role; see MATERIAL_ROLE_* below
	float noise_scale;         // world-space noise frequency multiplier for the role's noise fn
	float noise_warp;          // domain-warp strength applied before the role's noise fn evaluates
	float noise_sharpness;     // cell/edge crispness produced by the role's noise fn
	vec4  uv_scale_offset;     // xy: uv scale, zw: uv offset
	vec4  palette[4];          // four vec3s for cosine color palette
	vec4  base_color_factor;   // RGBA base-color multiply (glTF baseColorFactor) -- see MaterialData.hpp
	vec4  emissive_factor;     // RGB emissive multiply (glTF emissiveFactor) -- see MaterialData.hpp
	vec4  procedural_shading;  // x: macro/procedural normal blend strength; yzw: role-specific tuning
};

// Mirrors brassica::MaterialRole (MaterialData.hpp). Standard (0) must stay the default so every
// already-registered material (glTF imports, bark/rock/flower/etc.) renders through the
// untouched PBR path unless explicitly opted into a procedural role.
const uint MATERIAL_ROLE_STANDARD = 0u;
const uint MATERIAL_ROLE_PROCEDURAL_LEAF = 1u;
const uint MATERIAL_ROLE_PROCEDURAL_BARK = 2u;
const uint MATERIAL_ROLE_PROCEDURAL_ROCK = 3u;
const uint MATERIAL_ROLE_PROCEDURAL_NEEDLE = 4u;
const uint MATERIAL_ROLE_PROCEDURAL_CACTUS = 5u;

// Binding 6: the generic bindless storage-buffer catalog (PhysicalRegistry::
// AssignAndWriteBindlessBufferIndex, storageBufferBinding) -- every buffer with eStorageBuffer
// usage gets an index here automatically once the bindless set exists, independent of any
// explicit NodeContext::BufferIndex<K>() call on the C++ side. No single GLSL type fits every
// buffer that lands in this catalog, so each caller declares its own typed alias of the same
// binding number, same idiom as the image2D format aliases on binding 3 (bindless.glsl) --
// MaterialBufferArray is the first one. Previously dormant: this binding existed in C++ with zero
// GLSL declaration and zero callers (see Brassica GLTF pipeline Stage 1's own notes) -- this is
// the first thing that actually reads it.
layout(set = 1, binding = 6, std430) readonly buffer MaterialBufferArray {
	MaterialData data[];
}
uMaterialBuffers[];

/**
 * @brief Helper function to evaluate cosine color palette: a + b * cos(2*PI*(c*t + d))
 */
vec3 evaluateCosinePalette(MaterialData mat, float t) {
	const float TWO_PI = 6.283185307179586;
	return mat.palette[0].xyz + mat.palette[1].xyz * cos(TWO_PI * (mat.palette[2].xyz * t + mat.palette[3].xyz));
}

// The functions below are material_system.glsl's first real PBR consumers (previously only
// evaluateCosinePalette existed -- see Brassica's own notes on this system being "a speculative
// first pass for a few overlapping future goals," not a finished contract). Written for glTF
// materials specifically: texture_ids.y (normal) is parsed and stored by the loader but
// deliberately not sampled here -- tangent-space normal mapping needs a per-vertex tangent
// attribute GltfVertex doesn't carry yet, so sampling it would need a basis with no real tangent
// behind it. texture_ids.w (emissive) is likewise parsed but not composited into the deferred
// G-buffer yet -- glTF's emissive is additive on top of lit shading, and outPosition.w's existing
// "emissivity" contract (foliage.frag/entity.frag) looks like a lit/unlit bypass flag, not an
// additive-color channel; wiring a texture into a flag it wasn't designed for without checking
// deferred.frag's real contract would risk silently being wrong rather than just incomplete.

vec4 evaluateGltfBaseColor(MaterialData mat, vec2 uv) {
	vec4 color = mat.base_color_factor;
	if (mat.texture_ids.x != UNBOUND_TEXTURE_ID) {
		color *= texture(
			sampler2D(uTextures2D[nonuniformEXT(mat.texture_ids.x)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]),
			uv
		);
	}
	return color;
}

// .x = metallic, .y = roughness -- glTF packs both into one texture (G = roughness, B = metallic)
// by convention; texture_ids.z holds that single packed texture's bindless index (see
// GltfModel's own comment on why a separate occlusion-only texture can't also fit this one slot).
vec2 evaluateGltfMetallicRoughness(MaterialData mat, vec2 uv) {
	float metallic = mat.color_tint_offset.z;
	float roughness = mat.color_tint_offset.y;
	if (mat.texture_ids.z != UNBOUND_TEXTURE_ID) {
		vec4 packedSample = texture(
			sampler2D(uTextures2D[nonuniformEXT(mat.texture_ids.z)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]),
			uv
		);
		metallic *= packedSample.b;
		roughness *= packedSample.g;
	}
	return vec2(metallic, roughness);
}

// R channel of the same packed texture_ids.z slot the metallic/roughness function reads --
// correct exactly when glTF's occlusionTexture and metallicRoughnessTexture reference the same
// image, which GltfModel's loader requires (see its own comment); 1.0 (no occlusion) otherwise.
float evaluateGltfOcclusion(MaterialData mat, vec2 uv) {
	if (mat.texture_ids.z != UNBOUND_TEXTURE_ID) {
		return texture(
			sampler2D(uTextures2D[nonuniformEXT(mat.texture_ids.z)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]),
			uv
		)
			.r;
	}
	return 1.0;
}

#endif // HELPERS_MATERIAL_SYSTEM_GLSL
