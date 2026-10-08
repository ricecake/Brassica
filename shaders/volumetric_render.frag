#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform VolumetricRenderPushConstants {
	uvec4 cascadeScatteringIdx;  // x, y, z = sampled indices for cascade 0, 1, 2 scattering
	uvec4 cascadeExtinctionIdx;  // x, y, z = sampled indices for cascade 0, 1, 2 extinction
	uint  gPositionIndex;
	uint  gDepthIndex;
	uint  hdrColorIndex;
	float maxDistance;
} push;

void main() {
	vec4 currentHdr = SAMPLE_NEAREST(push.hdrColorIndex, inUV);
	vec4 posSample = SAMPLE_NEAREST(push.gPositionIndex, inUV);

	vec3 relPos = posSample.rgb;
	float rayDist = length(relPos);

	if (rayDist <= 0.001) {
		outColor = currentHdr;
		return;
	}

	// Cascade ranges
	float cascadeRanges[3] = float[3](50.0, 200.0, 800.0);

	vec3 accumulatedScattering = vec3(0.0);
	float accumulatedTransmittance = 1.0;

	// Sample froxel cascades along the ray
	vec3 viewDir = relPos / rayDist;
	int numSteps = 16;
	float stepSize = min(rayDist, cascadeRanges[2]) / float(numSteps);

	for (int i = 0; i < numSteps; ++i) {
		float t = (float(i) + 0.5) * stepSize;
		vec3 p = viewDir * t;

		// Determine cascade index
		int c = 0;
		if (t > cascadeRanges[0]) c = 1;
		if (t > cascadeRanges[1]) c = 2;

		uint scatIdx = push.cascadeScatteringIdx[c];
		uint extIdx = push.cascadeExtinctionIdx[c];

		if (scatIdx == 0xFFFFFFFFu || extIdx == 0xFFFFFFFFu) continue;

		float maxDist = cascadeRanges[c];
		vec3 uv3d = vec3(
			p.x / (maxDist * 2.0) + 0.5,
			p.y / (maxDist * 2.0) + 0.5,
			p.z / maxDist
		);

		if (uv3d.x < 0.0 || uv3d.x > 1.0 || uv3d.y < 0.0 || uv3d.y > 1.0 || uv3d.z < 0.0 || uv3d.z > 1.0) continue;

		vec4 scatSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(scatIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uv3d, 0.0);
		vec4 extSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(extIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uv3d, 0.0);

		vec3 scat = scatSample.rgb;
		float ext = extSample.r;

		float stepTransmittance = exp(-ext * stepSize);
		accumulatedScattering += scat * accumulatedTransmittance * stepSize;
		accumulatedTransmittance *= stepTransmittance;

		if (accumulatedTransmittance < 0.01) break;
	}

	vec3 finalColor = currentHdr.rgb * accumulatedTransmittance + accumulatedScattering;
	outColor = vec4(finalColor, currentHdr.a);
}
