#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform VolumetricRenderPushConstants {
	vec4  camForward;           // xyz = camera forward
	vec4  camUp;                // xyz = camera up
	vec4  camRight;             // xyz = camera right
	vec4  fovAspect;            // x = tanHalfFov, y = aspect ratio
	uvec4 cascadeScatteringIdx;  // x, y, z = sampled indices for cascade 0, 1, 2 scattering
	uvec4 cascadeExtinctionIdx;  // x, y, z = sampled indices for cascade 0, 1, 2 extinction
	uint  gPositionIndex;
	uint  gDepthIndex;
	uint  hdrColorIndex;
	float maxDistance;
} push;

vec4 sampleFroxelCascade(int c, vec3 relPos, out float outTransmittance) {
	float cascadeRanges[3] = float[3](200.0, 1600.0, 12800.0);
	float maxDist = cascadeRanges[c];

	uint scatIdx = push.cascadeScatteringIdx[c];
	uint extIdx = push.cascadeExtinctionIdx[c];

	if (scatIdx == 0xFFFFFFFFu || extIdx == 0xFFFFFFFFu) {
		outTransmittance = 1.0;
		return vec4(0.0);
	}

	// Project relative position onto camera basis
	float viewZ = dot(relPos, push.camForward.xyz);
	float viewX = dot(relPos, push.camRight.xyz);
	float viewY = dot(relPos, push.camUp.xyz);

	if (viewZ <= 0.001) {
		outTransmittance = 1.0;
		return vec4(0.0);
	}

	float tanHalfFov = push.fovAspect.x;
	float aspect = push.fovAspect.y;

	float frustumWidthAtZ = 2.0 * viewZ * aspect * tanHalfFov;
	float frustumHeightAtZ = 2.0 * viewZ * tanHalfFov;

	vec3 uv3d = vec3(
		clamp(viewX / frustumWidthAtZ + 0.5, 0.0, 1.0),
		clamp(viewY / frustumHeightAtZ + 0.5, 0.0, 1.0),
		clamp(viewZ / maxDist, 0.0, 1.0)
	);

	vec4 scatSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(scatIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);
	vec4 extSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(extIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);

	outTransmittance = extSample.r;
	return scatSample;
}

void main() {
	vec4 currentHdr = SAMPLE_NEAREST(push.hdrColorIndex, inUV);
	vec4 posSample = SAMPLE_NEAREST(push.gPositionIndex, inUV);

	vec3 relPos = posSample.rgb;
	float rayDist = length(relPos);

	if (rayDist <= 0.01) {
		outColor = currentHdr;
		return;
	}

	float cascadeRanges[3] = float[3](200.0, 1600.0, 12800.0);

	// Select cascade based on ray distance and smooth transition
	vec3 fogScattering = vec3(0.0);
	float fogTransmittance = 1.0;

	if (rayDist <= cascadeRanges[0]) {
		float t0 = 1.0;
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		fogScattering = s0.rgb;
		fogTransmittance = t0;
	} else if (rayDist <= cascadeRanges[1]) {
		float blend = smoothstep(cascadeRanges[0] * 0.8, cascadeRanges[0], rayDist);
		float t0 = 1.0, t1 = 1.0;
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);

		fogScattering = mix(s0.rgb, s1.rgb, blend);
		fogTransmittance = mix(t0, t1, blend);
	} else {
		float blend = smoothstep(cascadeRanges[1] * 0.8, cascadeRanges[1], rayDist);
		float t1 = 1.0, t2 = 1.0;
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);
		vec4 s2 = sampleFroxelCascade(2, relPos, t2);

		fogScattering = mix(s1.rgb, s2.rgb, blend);
		fogTransmittance = mix(t1, t2, blend);
	}

	vec3 finalColor = currentHdr.rgb * fogTransmittance + fogScattering;
	outColor = vec4(finalColor, currentHdr.a);
}
