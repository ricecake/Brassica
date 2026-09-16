#ifndef BRASSICA_LIGHTING_GLSL
#define BRASSICA_LIGHTING_GLSL

#include "bindless.glsl"

const int LIGHT_FLAG_CASTS_SHADOW = 1;
const int LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
const int LIGHT_FLAG_CAMERA_RELATIVE = 4;
const int LIGHT_FLAG_CLOUD_EMISSIVE = 8;

const int LIGHT_TYPE_POINT = 0;
const int LIGHT_TYPE_DIRECTIONAL = 1;
const int LIGHT_TYPE_SPOT = 2;
const int LIGHT_TYPE_EMISSIVE = 3;
const int LIGHT_TYPE_FLASH = 4;

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

layout(std430, set = 0, binding = 2) readonly buffer LightsBuffer {
	uint  uLightCount;
	uint  uLightPad[3];
	Light uLights[];
};

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
