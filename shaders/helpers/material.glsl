#ifndef HELPERS_MATERIAL_GLSL
#define HELPERS_MATERIAL_GLSL

// Shaped after lygia's lighting/material.glsl + lighting/material/new.glsl pattern (a Material
// struct plus a materialDefault() constructor) rather than including lygia's directly -- that one
// also carries raymarching-specific fields (sdf/valid), folds position/normal into the struct, and
// gates clear-coat/iridescence/subsurface behind #defines, none of which fits how this engine's
// deferred/water shaders already pass frag_pos/normal as separate arguments. Scoped to only what
// evaluate_brdf actually consumes today.
//
// The point of routing every material-property call site through this one struct: adding a new
// property later (microfacet glint density, subsurface thickness, whatever) means adding a field
// here with a default in materialDefault(), then using it inside evaluate_brdf's body -- every
// existing call site keeps passing the struct through unchanged, nothing else needs to change.
struct Material {
	vec3  albedo;
	float roughness;
	float metallic;
	float ao;
};

Material materialDefault() {
	return Material(vec3(1.0), 0.7, 0.0, 1.0);
}

#endif // HELPERS_MATERIAL_GLSL
