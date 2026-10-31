#ifndef BRASSICA_FOLIAGE_ARCHETYPE_GLSL
#define BRASSICA_FOLIAGE_ARCHETYPE_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

// Shared between foliage_bake.mesh (the sole writer) and whatever Stage 3 render pass becomes the
// reader: one struct definition included by both, instead of two hand-copied ones drifting apart
// the way terrain.task/terrain.mesh's TaskPayload used to (see terrain_payload.glsl's own
// comment) -- std430 layout rules only need to agree with themselves, not with any C++ mirror,
// since nothing outside the GPU ever touches this data.
//
// Archetype slots are fixed-capacity (kFoliageArchetypeMaxVerts/kFoliageArchetypeMaxTris per slot
// -- comfortably above every tree/bush variant's real vertex/triangle count, see
// foliage_emit.glsl's call sites), so a slot's base offset is a plain multiply, no separate
// offset table needed; FoliageArchetypeMeta only needs to carry how much of that fixed slot is
// actually used.
//
// Positions/normals are local space: baked with basePos = vec3(0.0), sway = vec3(0.0), fade = 1.0
// (see foliage_bake.mesh) so every per-instance transform -- world position, rotation, biome
// scale/tint, wind sway, LOD fade -- is applied at draw time instead of baked in, matching the
// archetype-palette design (shape is shared, everything else is per-instance).
const uint kFoliageArchetypeMaxVerts = 128u;
const uint kFoliageArchetypeMaxTris = 128u;

const uint kFoliageTreeSubtypeCount = 4u;   // pine, deciduous, palm, cactus -- matches getWhittakerTreeType
const uint kFoliageTreeVariantsPerSubtype = 2u;
const uint kFoliageTreeArchetypeCount = kFoliageTreeSubtypeCount * kFoliageTreeVariantsPerSubtype; // 8
const uint kFoliageBushArchetypeCount = 8u;
const uint kFoliageArchetypeCount = kFoliageTreeArchetypeCount + kFoliageBushArchetypeCount; // 16

// Tree archetypes occupy slots [0, kFoliageTreeArchetypeCount); bush archetypes occupy
// [kFoliageTreeArchetypeCount, kFoliageArchetypeCount).
uint foliageTreeArchetypeSlot(uint treeSubtype, uint variant) {
	return treeSubtype * kFoliageTreeVariantsPerSubtype + (variant % kFoliageTreeVariantsPerSubtype);
}

uint foliageBushArchetypeSlot(uint variant) {
	return kFoliageTreeArchetypeCount + (variant % kFoliageBushArchetypeCount);
}

struct FoliageArchetypeVertex {
	vec3 position; // local space, relative to the item's own base (basePos = 0 at bake time)
	vec3 normal;
	vec2 uv;
	vec4 color;
	uint materialType;
};

struct FoliageArchetypeMeta {
	uint vertexCount;
	uint triangleCount;
};

layout(buffer_reference, std430) buffer FoliageArchetypeVertexRef {
	FoliageArchetypeVertex vertices[];
};

layout(buffer_reference, std430) buffer FoliageArchetypeIndexRef {
	uvec3 triangles[];
};

layout(buffer_reference, std430) buffer FoliageArchetypeMetaRef {
	FoliageArchetypeMeta slots[];
};

#endif // BRASSICA_FOLIAGE_ARCHETYPE_GLSL
