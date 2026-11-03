#ifndef BRASSICA_FOLIAGE_INSTANCE_GLSL
#define BRASSICA_FOLIAGE_INSTANCE_GLSL

#include "bindless.glsl"

// Per-frame tree/bush instance list, collected by foliage.mesh (the same per-clump pass that
// still generates live geometry for every foliage type today -- see foliage.mesh's own comment
// at the append site) and meant to be consumed by a later indirect render pass that draws each
// instance from its baked archetype shape (foliage_archetype.glsl) instead of regenerating it.
// Rebuilt fresh every frame, unlike the archetype buffers -- this is "who's visible and where
// right now," not cached shape data.
//
// Position/rotation/fade are exactly what foliage.mesh already computes per item (basePos/angle/
// fade); length/width multipliers are deliberately NOT stored per instance -- they're global
// FoliageState sliders, applied uniformly to every instance at draw time instead, not something
// that varies per instance.
const uint kFoliageInstanceCapacity = 8192u;

struct FoliageInstanceData {
	vec3  relPosition; // camera-relative, valid only for the frame it was collected in
	float rotationAngle;
	float fade;
	uint  archetypeSlot;
};

// Matches VkDrawMeshTasksIndirectCommandEXT's layout exactly (groupCountX/Y/Z, all uint32) --
// groupCountX doubles as the live append counter (atomicAdd'd directly, see foliage.mesh), so
// the count an eventual indirect draw reads is already correct the moment collection finishes,
// with no separate copy step.
struct FoliageIndirectArgs {
	uint groupCountX;
	uint groupCountY;
	uint groupCountZ;
};

// Binding 6 aliases (see material_system.glsl's own comment on this binding) -- both buffers are
// eStorageBuffer-usage and already auto-registered by PhysicalRegistry::AssignAndWriteBindlessBufferIndex.
layout(set = 1, binding = 6, std430) buffer FoliageInstanceBufferArray {
	FoliageInstanceData instances[];
}
uFoliageInstanceBuffers[];

layout(set = 1, binding = 6, std430) buffer FoliageIndirectArgsBufferArray {
	FoliageIndirectArgs args;
}
uFoliageIndirectArgsBuffers[];

#endif // BRASSICA_FOLIAGE_INSTANCE_GLSL
