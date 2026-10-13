#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform VolumetricRenderPushConstants {
	vec4  scaledCamRight;        // xyz = camRight * aspect * tanHalfFov
	vec4  scaledCamUp;           // xyz = camUp * tanHalfFov
	uvec4 cascadeScatExtIdx01;   // x,y = C0 scat/ext, z,w = C1 scat/ext
	uvec4 cascadeScatExtIdx23;   // x,y = C2 scat/ext, z,w = C3 scat/ext
	uvec4 gBufferConfig;        // x = gPositionIndex, y = gDepthIndex, z = hdrColorIndex, w = uintBitsToFloat(maxDist)
} push;

vec4 sampleFroxelCascade(int c, vec3 relPos, out vec3 outTransmittance) {
	float cascadeRanges[4] = float[4](200.0, 1600.0, 12800.0, 51200.0);
	float maxDist = cascadeRanges[c];

	uvec2 scatExtIndices[4] = uvec2[4](
		push.cascadeScatExtIdx01.xy,
		push.cascadeScatExtIdx01.zw,
		push.cascadeScatExtIdx23.xy,
		push.cascadeScatExtIdx23.zw
	);

	uint scatIdx = scatExtIndices[c].x;
	uint extIdx = scatExtIndices[c].y;

	if (scatIdx == 0xFFFFFFFFu || extIdx == 0xFFFFFFFFu) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	vec3 normCamRight = normalize(push.scaledCamRight.xyz);
	vec3 normCamUp = normalize(push.scaledCamUp.xyz);
	vec3 normCamForward = normalize(cross(normCamUp, normCamRight));

	// Project relative position onto camera basis
	float viewZ = dot(relPos, normCamForward);
	float viewX = dot(relPos, normCamRight);
	float viewY = dot(relPos, normCamUp);

	if (viewZ <= 0.001) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	float frustumWidthAtZ = 2.0 * viewZ * length(push.scaledCamRight.xyz);
	float frustumHeightAtZ = 2.0 * viewZ * length(push.scaledCamUp.xyz);

	vec3 uv3d = vec3(
		clamp(viewX / frustumWidthAtZ + 0.5, 0.0, 1.0),
		clamp(0.5 - viewY / frustumHeightAtZ, 0.0, 1.0),
		clamp(length(relPos) / maxDist, 0.0, 1.0)
	);

	vec4 scatSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(scatIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);
	vec4 extSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(extIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);

	outTransmittance = extSample.rgb;
	return scatSample;
}

void main() {
	uint gPositionIndex = push.gBufferConfig.x;
	uint hdrColorIndex = push.gBufferConfig.z;
	float maxDistance = uintBitsToFloat(push.gBufferConfig.w);

	vec4 currentHdr = SAMPLE_NEAREST(hdrColorIndex, inUV);
	vec4 posSample = SAMPLE_NEAREST(gPositionIndex, inUV);

	vec3 relPos = posSample.rgb;
	float rayDist = length(relPos);

	if (rayDist <= 0.01) {
		// Empty sky/background pixel (no G-Buffer geometry rendered): reconstruct view ray direction
		vec2 clip = inUV * 2.0 - 1.0;
		vec4 viewRay4 = uInvProjMatrix * vec4(clip, 1.0, 1.0);
		vec3 viewDir = viewRay4.xyz / max(1e-4, viewRay4.w);
		vec3 worldRayDir = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

		// Local fog volume extends up to ~600m along background sky rays
		float skyFogDist = min(maxDistance, 600.0);
		relPos = worldRayDir * skyFogDist;
		rayDist = skyFogDist;
	}

	float cascadeRanges[4] = float[4](200.0, 1600.0, 12800.0, 51200.0);

	// Select cascade based on ray distance and smooth transition
	vec3 fogScattering = vec3(0.0);
	vec3 fogTransmittance = vec3(1.0);

	if (rayDist <= cascadeRanges[0]) {
		vec3 t0 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		fogScattering = s0.rgb;
		fogTransmittance = t0;
	} else if (rayDist <= cascadeRanges[1]) {
		float blend = smoothstep(cascadeRanges[0] * 0.8, cascadeRanges[0], rayDist);
		vec3 t0 = vec3(1.0), t1 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);

		fogScattering = mix(s0.rgb, s1.rgb, blend);
		fogTransmittance = mix(t0, t1, blend);
	} else if (rayDist <= cascadeRanges[2]) {
		float blend = smoothstep(cascadeRanges[1] * 0.8, cascadeRanges[1], rayDist);
		vec3 t1 = vec3(1.0), t2 = vec3(1.0);
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);
		vec4 s2 = sampleFroxelCascade(2, relPos, t2);

		fogScattering = mix(s1.rgb, s2.rgb, blend);
		fogTransmittance = mix(t1, t2, blend);
	} else {
		float blend = smoothstep(cascadeRanges[2] * 0.8, cascadeRanges[2], rayDist);
		vec3 t2 = vec3(1.0), t3 = vec3(1.0);
		vec4 s2 = sampleFroxelCascade(2, relPos, t2);
		vec4 s3 = sampleFroxelCascade(3, relPos, t3);

		fogScattering = mix(s2.rgb, s3.rgb, blend);
		fogTransmittance = mix(t2, t3, blend);
	}

	vec3 finalColor = currentHdr.rgb * fogTransmittance + fogScattering;
	outColor = vec4(finalColor, currentHdr.a);
}
