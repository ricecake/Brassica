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

#ifdef LIGHTS_BUFFER_WRITABLE
layout(std430, set = 0, binding = 2) buffer LightsBuffer {
	uint  uLightCount;
	uint  uLightPad[3];
	Light uLights[];
};
#else
layout(std430, set = 0, binding = 2) readonly buffer LightsBuffer {
	uint  uLightCount;
	uint  uLightPad[3];
	Light uLights[];
};
#endif

// Number of concurrent GPU-authored lightning strikes -- matches
// include/types/LightningStrike.hpp's kMaxLightningStrikes exactly (that header sizes the
// buffer allocation; this is the GLSL side's copy of the same constant, same convention as
// every other *PushConstants mirror in this codebase).
#define LIGHTNING_MAX_STRIKES 32

// Reserved uLights[] slot range for lightning flash lights -- static per-slot assignment,
// same idiom as particle_behavior.comp's firefly lights (16u + particleIdx): slots
// LIGHTNING_LIGHT_SLOT_BASE..+LIGHTNING_MAX_STRIKES-1, written by shaders/lightning_age.comp.
#define LIGHTNING_LIGHT_SLOT_BASE 256u

struct LightningStrikeSlot {
	vec4  originAndSeed; // xyz = cloud-side origin world pos, w = PRNG seed
	vec4  endPos;        // xyz = second cloud-side point, w = unused
	float lifetime;
	float maxLifetime;
	float _pad0;
	float _pad1;
};

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

struct Cluster {
	uint count;
	uint lightIndices[64];
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

#endif // BRASSICA_LIGHTING_GLSL
