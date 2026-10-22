#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "common.glsl"
#include "atmosphere/common.glsl"

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
	uint  skyViewIndex;
	uint  transmittanceIndex;
	float maxDistance;
} push;

float froxelDistToNorm(float dist, float maxDist, float alpha) {
	float normDist = clamp(dist / maxDist, 0.0, 1.0);
	return log(1.0 + normDist * (exp(alpha) - 1.0)) / alpha;
}

vec4 sampleFroxelCascade(int c, vec3 relPos, out vec3 outTransmittance) {
	float cascadeRanges[3] = float[3](300.0, 3000.0, 100000.0);
	float maxDist = cascadeRanges[c];

	uint scatIdx = push.cascadeScatteringIdx[c];
	uint extIdx = push.cascadeExtinctionIdx[c];

	if (scatIdx == 0xFFFFFFFFu || extIdx == 0xFFFFFFFFu) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	// Project relative position onto camera basis
	float viewZ = dot(relPos, push.camForward.xyz);
	float viewX = dot(relPos, push.camRight.xyz);
	float viewY = dot(relPos, push.camUp.xyz);

	if (viewZ <= 0.001) {
		outTransmittance = vec3(1.0);
		return vec4(0.0);
	}

	float tanHalfFov = push.fovAspect.x;
	float aspect = push.fovAspect.y;

	float frustumWidthAtZ = 2.0 * viewZ * aspect * tanHalfFov;
	float frustumHeightAtZ = 2.0 * viewZ * tanHalfFov;

	float rayLen = length(relPos);
	float zNorm = froxelDistToNorm(rayLen, maxDist, 3.5);

	vec3 uv3d = vec3(
		clamp(viewX / frustumWidthAtZ + 0.5, 0.0, 1.0),
		clamp(0.5 - viewY / frustumHeightAtZ, 0.0, 1.0),
		clamp(zNorm, 0.0, 1.0)
	);

	vec4 scatSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(scatIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);
	vec4 extSample = textureLod(sampler3D(uTextures3D[nonuniformEXT(extIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv3d, 0.0);

	outTransmittance = extSample.rgb;
	return scatSample;
}

void main() {
	vec4 currentHdr = SAMPLE_NEAREST(push.hdrColorIndex, inUV);
	vec4 posSample = SAMPLE_NEAREST(push.gPositionIndex, inUV);

	vec3 relPos = posSample.rgb;
	float rayDist = length(relPos);
	bool hasSurface = posSample.a >= 0.01 && rayDist > 0.01;

	vec3 worldRayDir;
	if (hasSurface) {
		worldRayDir = relPos / max(1e-3, rayDist);
	} else {
		// Empty sky/clouds/background: reconstruct view ray direction and sample maxDistance
		vec2 clip = inUV * 2.0 - 1.0;
		vec4 viewRay4 = uInvProjMatrix * vec4(clip, 1.0, 1.0);
		vec3 viewDir = viewRay4.xyz / max(1e-4, viewRay4.w);
		worldRayDir = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

		rayDist = push.maxDistance;
		relPos = worldRayDir * rayDist;
	}

	float cascadeRanges[3] = float[3](300.0, 3000.0, 100000.0);

	// Select cascade based on ray distance and smooth transition
	vec3 fogScattering = vec3(0.0);
	vec3 fogTransmittance = vec3(1.0);

	if (rayDist <= cascadeRanges[0]) {
		vec3 t0 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		fogScattering = s0.rgb;
		fogTransmittance = t0;
	} else if (rayDist <= cascadeRanges[1]) {
		float blend = smoothstep(cascadeRanges[0] * 0.7, cascadeRanges[0], rayDist);
		vec3 t0 = vec3(1.0), t1 = vec3(1.0);
		vec4 s0 = sampleFroxelCascade(0, relPos, t0);
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);

		fogScattering = mix(s0.rgb, s1.rgb, blend);
		fogTransmittance = mix(t0, t1, blend);
	} else {
		float blend = smoothstep(cascadeRanges[1] * 0.7, cascadeRanges[1], rayDist);
		vec3 t1 = vec3(1.0), t2 = vec3(1.0);
		vec4 s1 = sampleFroxelCascade(1, relPos, t1);
		vec4 s2 = sampleFroxelCascade(2, relPos, t2);

		fogScattering = mix(s1.rgb, s2.rgb, blend);
		fogTransmittance = mix(t1, t2, blend);
	}

	// Underwater extinction when camera is submerged
	float camAltKM = uCameraPosition.y / 1000.0;
	float waterLevelKM = u_waterLevel / 1000.0;
	if (camAltKM < waterLevelKM) {
		vec3 waterExtCoeff = vec3(0.45, 0.08, 0.03); // Red attenuated fastest
		float distKM = rayDist / 1000.0;
		vec3 waterTrans = exp(-waterExtCoeff * distKM);
		fogTransmittance *= waterTrans;
		fogScattering *= waterTrans;
	}

	// Converge distant haze toward SkyViewLUT radiance
	if (push.skyViewIndex > 0u && push.skyViewIndex != 0xFFFFFFFFu && camAltKM >= waterLevelKM) {
		float fogOpacity = 1.0 - clamp(dot(fogTransmittance, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
		vec3 skyColor = sampleSkyView(push.skyViewIndex, worldRayDir);
		fogScattering = mix(fogScattering, skyColor * fogOpacity, fogOpacity * u_skyConvergence);
	}

	vec3 finalColor = currentHdr.rgb * fogTransmittance + fogScattering;
	outColor = vec4(finalColor, currentHdr.a);
}
