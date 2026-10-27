#ifndef BRASSICA_CBT_GLSL
#define BRASSICA_CBT_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common.glsl"

#define CBT_MAX_DEPTH 28u
#define CBT_INVALID_PAGE 0xFFFFFFFFu

struct Bisector {
	uint64_t id;
	uint     depth;
	uint     rootIdx;
	uint     twin;
	uint     next;
	uint     prev;
	uint     cachePage;
	uint     padding;
	vec2     v0;
	vec2     v1;
	vec2     v2;
};

struct CBTCommand {
	uint bisectorIdx;
	uint type; // 0 = none, 1 = split, 2 = merge
};

struct PageTableEntry {
	uint64_t bisectorID;
	uint     pageIndex;
	uint     flags; // 1 = valid, 0 = free
};

struct GeometryCacheHeader {
	vec2  minMax;
	float avgHeight;
	float padding;
};

// Longest-edge bisection of a right triangle v0-v1-v2 (v0-v1 is hypotenuse)
void cbt_bisect_triangle(vec2 v0, vec2 v1, vec2 v2, uint childBit, out vec2 outV0, out vec2 outV1, out vec2 outV2) {
	vec2 m = (v0 + v1) * 0.5;
	if (childBit == 0u) {
		outV0 = m;
		outV1 = v2;
		outV2 = v0;
	} else {
		outV0 = m;
		outV1 = v1;
		outV2 = v2;
	}
}

// Decodes bisector 2D vertices from root triangle and 64-bit bisector ID
void cbt_decode_bisector_vertices(
	uint64_t id,
	uint depth,
	vec2 rootV0,
	vec2 rootV1,
	vec2 rootV2,
	out vec2 outV0,
	out vec2 outV1,
	out vec2 outV2
) {
	outV0 = rootV0;
	outV1 = rootV1;
	outV2 = rootV2;

	for (uint i = 0u; i < depth; ++i) {
		uint shift = depth - 1u - i;
		uint childBit = uint((id >> shift) & 1UL);
		cbt_bisect_triangle(outV0, outV1, outV2, childBit, outV0, outV1, outV2);
	}
}

// Parabolic planet curvature drop-off calculation: z_drop = -d^2 / (2 * R)
float cbt_apply_fake_curvature(vec2 worldXZ) {
	vec2 rel = worldXZ - uCameraPosition.xz;
	float distSq = dot(rel, rel);
	return -(distSq / (2.0 * FAKE_PLANET_RADIUS));
}

#endif // BRASSICA_CBT_GLSL
