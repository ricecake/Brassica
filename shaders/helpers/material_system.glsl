#ifndef HELPERS_MATERIAL_SYSTEM_GLSL
#define HELPERS_MATERIAL_SYSTEM_GLSL

const uint UNBOUND_TEXTURE_ID = 0xFFFFFFFFu;

/**
 * @brief GLSL MaterialData matching std430 C++ MaterialData struct layout.
 */
struct MaterialData {
	uvec4 texture_ids;        // x: albedo, y: normal, z: occlusion/roughness/metallic, w: emission
	vec4  color_tint_offset;  // x: color tint offset, y: roughness, z: metallic, w: emission strength
	uint  shading_bitmask;    // shading control flow bitmask
	vec4  uv_scale_offset;    // xy: uv scale, zw: uv offset
	vec4  palette[4];         // four vec3s for cosine color palette
};

/**
 * @brief Helper function to evaluate cosine color palette: a + b * cos(2*PI*(c*t + d))
 */
vec3 evaluateCosinePalette(MaterialData mat, float t) {
	const float TWO_PI = 6.283185307179586;
	return mat.palette[0].xyz + mat.palette[1].xyz * cos(TWO_PI * (mat.palette[2].xyz * t + mat.palette[3].xyz));
}

#endif // HELPERS_MATERIAL_SYSTEM_GLSL
