#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	vec3  waterColor;
	float waterLevel;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
} params;

// Real alpha blending over whatever DeferredNode already wrote (this pass runs at
// graph::Phase::Late, strictly after it) -- outputting alpha 0 wherever there's nothing to tint
// lets the blend equation leave the destination untouched, so there's no need to discard.
void main() {
	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, inUV);
	// albedo.a < 0.01 is the same "no terrain rendered here" sentinel deferred.frag reads --
	// without this check, GBufferPosition's cleared-to-zero value at sky/background pixels
	// would read as "at the water level", tinting the sky.
	if (albedo.a < 0.01) {
		outColor = vec4(0.0);
		return;
	}

	float terrainHeight = SAMPLE_NEAREST(params.gPositionIndex, inUV).y;
	float depthBelowWater = params.waterLevel - terrainHeight;
	if (depthBelowWater <= 0.0) {
		outColor = vec4(0.0);
		return;
	}

	// Matches terrain.mesh's own inline water-tint formula, kept here as a real depth-composited
	// blend instead of a per-vertex color hack.
	float alpha = clamp(0.35 + depthBelowWater * 0.04, 0.35, 0.85);
	outColor = vec4(params.waterColor, alpha);
}
