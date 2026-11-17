#ifndef BRASSICA_LIGHTING_GLSL
#define BRASSICA_LIGHTING_GLSL

#include "bindless.glsl"
#ifndef LIGHTING_TYPES
#define LIGHTING_TYPES
const int LIGHT_FLAG_CASTS_SHADOW = 1;
const int LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
const int LIGHT_FLAG_CAMERA_RELATIVE = 4;
const int LIGHT_FLAG_CLOUD_EMISSIVE = 8;

const int LIGHT_TYPE_POINT = 0;
const int LIGHT_TYPE_DIRECTIONAL = 1;
const int LIGHT_TYPE_SPOT = 2;
const int LIGHT_TYPE_EMISSIVE = 3;
const int LIGHT_TYPE_FLASH = 4;
#endif

layout(std140, set = 0, binding = 1) uniform LightingUBO {
	uint  uNumLights;
	float uDayTime;
	float uNightFactor;
	float uWorldScale;

	vec4  uAmbientLight;    // xyz = color
	vec4  uLightningColor;  // xyz = color, w = pulse

	float uSkyExposure;
	float uStarExposure;
	float uTerrainExposure;
	float uPaddingExposure;

	vec4  uSHCoeffs[81];
};

struct Light {
	vec3  position;
	float intensity;
	vec3  color;
	int   type;
	vec3  direction;
	float innerCutoff;
	float outerCutoff;
	int   flags;
	float _pad0;
	float _pad1;
};

// Split from the old single fixed-binding LightsBuffer into two bindless-indexed buffers (the
// generic bindless storage-buffer catalog, binding 6 -- see material_system.glsl's identical
// comment on this binding's "every caller declares its own typed alias" convention). Each
// caller passes its own index (NodeContext::BufferIndex<CpuLightsBuffer/GpuLightsBuffer>(), a
// push-constant field) and accesses uCpuLightsBuffers[nonuniformEXT(idx)]/uGpuLightsBuffers[
// nonuniformEXT(idx)] instead of a plain global -- see include/resources/core.hpp's identical
// comment for why the split exists.
//
// CpuLightsBuffer is always readonly from every shader's perspective -- only LightManager's own
// host upload ever writes it.
layout(std430, set = 1, binding = 6) readonly buffer CpuLightsBufferArray {
	uint  count;
	uint  pad[3];
	Light lights[];
}
uCpuLightsBuffers[];

#ifdef GPU_LIGHTS_BUFFER_WRITABLE
layout(std430, set = 1, binding = 6) buffer GpuLightsBufferArray {
	uint  count;
	uint  dynamicClaimCounter;
	uint  pad[2];
	Light lights[];
}
uGpuLightsBuffers[];
#else
layout(std430, set = 1, binding = 6) readonly buffer GpuLightsBufferArray {
	uint  count;
	uint  dynamicClaimCounter;
	uint  pad[2];
	Light lights[];
}
uGpuLightsBuffers[];
#endif

// GLSL-side copies of include/constants.h's Lighting::MaxGpuLights/GpuLightDynamicCapacity --
// GpuLightsResetNode (gpu_lights_reset.comp) sizes its clear dispatch from these; fireflies
// (particle_behavior.comp) clamp their atomic claim to MAX_GPU_LIGHTS_DYNAMIC.
#define MAX_GPU_LIGHTS 512u
#define MAX_GPU_LIGHTS_DYNAMIC 480u

// Number of concurrent GPU-authored lightning strikes -- matches
// include/types/LightningStrike.hpp's kMaxLightningStrikes exactly (that header sizes the
// buffer allocation; this is the GLSL side's copy of the same constant, same convention as
// every other *PushConstants mirror in this codebase).
#define LIGHTNING_MAX_STRIKES 32

// Reserved GpuLightsBuffer slot range for lightning flash lights -- static per-slot assignment,
// written by shaders/lightning_age.comp. Matches include/constants.h's
// Lighting::GpuLightDynamicCapacity: [0, LIGHTNING_LIGHT_SLOT_BASE) is fireflies' atomic-claimed
// range (particle_behavior.comp), [LIGHTNING_LIGHT_SLOT_BASE, LIGHTNING_LIGHT_SLOT_BASE +
// LIGHTNING_MAX_STRIKES) is this reserved range.
#define LIGHTNING_LIGHT_SLOT_BASE 480u

// LIGHTNING_FLAG_GROUND: cloud-to-ground strike (endPos is the ground point) instead of
// cloud-to-cloud (endPos is a second cloud-side point). LIGHTNING_FLAG_PENDING_GROUND:
// climate_ca.comp just claimed this as a ground strike but has no terrain-height source of its
// own (see its trigger comment) -- endPos.y is a placeholder until lightning_age.comp resolves
// it via terrain/generation.glsl and clears the bit.
#define LIGHTNING_FLAG_GROUND 1u
#define LIGHTNING_FLAG_PENDING_GROUND 2u

struct LightningStrikeSlot {
	vec4  originAndSeed; // xyz = flat-world origin, w = PRNG seed
	vec4  endPos;        // xyz = flat-world end point, w = unused
	vec4  colorAndPeak;  // rgb = base color, a = per-strike peak brightness
	float lifetime;
	float maxLifetime;
	uint  flags;
	uint  _pad;
};

// Shared age -> brightness flicker envelope for a strike-pool slot: fast attack, then a decaying
//100Hz-ish flicker. lightning_age.comp, lightning.mesh and cloud_raymarch.comp's strike-glow
// selection all call this instead of each keeping their own copy (they used to drift
// independently). `lifetime` is the slot's *remaining* countdown (maxLifetime -> 0), so
// normalizedAge runs 0 (birth) -> 1 (death). Uses the literal 3.14159265, not the PI symbol:
// this file is included before common.glsl in at least one consumer (cloud_raymarch.comp), so a
// bare PI reference would be used before its #define is textually visible.
float lightningEnvelope(float lifetime, float maxLifetime) {
	float normalizedAge = 1.0 - clamp(lifetime / max(maxLifetime, 0.0001), 0.0, 1.0);
	return max(0.0, sin(normalizedAge * 3.14159265) * (0.8 + 0.2 * sin(lifetime * 50.0)));
}

// climate_ca.comp claims a slot (atomicAdd(uStrikeRingCounter, 1u) % LIGHTNING_MAX_STRIKES) and
// writes a fresh strike when weather conditions trigger one; lightning_age.comp ages every slot
// each frame and writes its reserved uLights[] entry; lightning.task reads (read-only) to plot
// the bolt polyline and cull dead slots.
#ifdef LIGHTNING_STRIKE_POOL_WRITABLE
layout(std430, set = 0, binding = 7) buffer LightningStrikePool {
	uint                 uStrikeRingCounter;
	uint                 uStrikePad[3];
	LightningStrikeSlot  uStrikes[];
};
#else
layout(std430, set = 0, binding = 7) readonly buffer LightningStrikePool {
	uint                 uStrikeRingCounter;
	uint                 uStrikePad[3];
	LightningStrikeSlot  uStrikes[];
};
#endif

// Mirrors C++ ClusterGPU (Light.hpp) exactly, including its 320-byte alignas(16) padding --
// cpuLightIndices/cpuCount are a small, separate list from lightIndices/count (GPU-sourced) --
// see cluster_light_assignment.comp's own comment for why.
struct Cluster {
	uint count;
	uint lightIndices[64];
	uint cpuCount;
	uint cpuLightIndices[8];
	uint padding[3];
};

#ifdef CLUSTER_GRID_WRITABLE
layout(std430, set = 0, binding = 3) buffer ClusterGridBuffer {
	Cluster uClusters[];
};
#else
layout(std430, set = 0, binding = 3) readonly buffer ClusterGridBuffer {
	Cluster uClusters[];
};
#endif

// Computes the 1D cluster index for a given world-space position (16x9x24 grid, matching
// cluster_light_assignment.comp's own dispatch dims and include/constants.h's GridX/Y/Z).
// Moved here from clustered_lighting.glsl (which still gets it transitively via that file's own
// #include "lighting.glsl") so any shader that just needs to look lights up in uClusters[] --
// e.g. shaders/volumetric_froxel_gen.comp -- doesn't have to pull in clustered_lighting.glsl's
// whole PBR/GTAO evaluation machinery just for this one function.
uint getClusterIndex(vec3 frag_pos) {
	vec4 view_pos = uViewMatrix * vec4(frag_pos, 1.0);
	float z_val = -view_pos.z;

	float z_near = uNearPlane;
	float z_far = uFarPlane;

	int z_slice = int(clamp(log(max(z_val, 0.001) / z_near) / log(z_far / z_near) * 24.0, 0.0, 23.0));

	vec4 clip_pos = uProjMatrix * view_pos;
	vec3 ndc_pos = clip_pos.xyz / max(0.0001, clip_pos.w);
	vec2 screen_uv = ndc_pos.xy * 0.5 + 0.5;

	int x_slice = int(clamp(screen_uv.x * 16.0, 0.0, 15.0));
	int y_slice = int(clamp(screen_uv.y * 9.0, 0.0, 8.0));

	return uint(x_slice + y_slice * 16 + z_slice * 16 * 9);
}

#endif // BRASSICA_LIGHTING_GLSL
