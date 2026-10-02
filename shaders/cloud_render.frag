#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "lighting.glsl"
#include "atmosphere/common.glsl"
#include "helpers/octahedral.glsl"
#include "lygia/generative/psrdnoise.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CloudRenderPushConstants {
	vec4  sunDir;
	vec4  sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	uvec4 cascadeSampledIdx; // x, y, z = sampled bindless indices for cascades 0, 1, 2
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  hdrColorIndex;
	uint  transmittanceIndex;
	uint  skyViewIndex;
	uint  weatherBiomeIndex;
	float cloudOpacity;
} push;

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

vec3 calculateCloudColor(
	float t_cloud,
	vec3  worldRay,
	float worldScale,
	float planetRadius,
	float cloudAlt,
	vec3  sunDir,
	vec3  sunRadiance,
	vec3  skyRadiance
) {
	// // Sample from the 3 volume texture cascades (minimally populated sample)

	vec3 p_cloud = uCameraPosition.xyz + worldRay * (t_cloud * 1000.0 * worldScale);

	vec3 planetCenter = vec3(0.0, -FAKE_PLANET_RADIUS, 0.0);
	vec3 surfaceDir = normalize(p_cloud - planetCenter);
	vec2 weatherUV = directionToOctahedralUV(surfaceDir);
	vec4 cascade0Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.x, p_cloud/20000);
	vec4 cascade1Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, p_cloud/80000);
	vec4 cascade2Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, p_cloud/320000);
	vec3 cloudTex = vec3(cascade0Sample.r, cascade1Sample.r, cascade2Sample.r);

	float weatherCloudDensity = 0.5;
	float rainfall = 0.0;

	if (push.weatherBiomeIndex > 0u) {
		vec4 weatherSample = SAMPLE_LINEAR(push.weatherBiomeIndex, weatherUV);
		weatherCloudDensity = clamp(weatherSample.g, 0.0, 1.0);
		cloudTex *= weatherCloudDensity;
	}

	vec3  T_cloud = max(getTransmittance(planetRadius + cloudAlt, sunDir.y), vec3(0.001));
	float cloudPhase = mix(0.2, 1.0, pow(max(0.0, dot(worldRay, sunDir)), 3.0));

	float baseOpacity = push.cloudOpacity > 0.0 ? push.cloudOpacity : 0.0125;
	float opacity = baseOpacity * (0.6 + 1.4 * length(cloudTex));

	vec3  cloudLighting = (T_cloud * sunRadiance * cloudPhase * 5.0) + (skyRadiance * 0.5);

	// Darken clouds where there is rainfall
	float rainDarkening = mix(1.0, 0.35, rainfall);
	cloudLighting *= rainDarkening;

	vec3  cloudColor = cloudLighting * opacity * 15.0*cloudTex;

	float opticalDepthFade = exp(-t_cloud * 0.0025);
	cloudColor *= opticalDepthFade;

	// Proximity fade: As the camera gets closer to the layer, the area near the camera becomes transparent
	// to avoid directly seeing how thin the clouds are when passing through.
	float proximityFade = smoothstep(0.1, 3.0, t_cloud);
	cloudColor *= proximityFade;

	return cloudColor;
}

void main() {
	vec3 currentRadiance = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;
	vec4 albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3 relPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;

	float worldScale = 1.0;
	float waterLevelKM = u_waterLevel / 1000.0;
	float camAltKM = uCameraPosition.y / (1000.0 * worldScale);

	// When submerged underwater, skip cloud cloud layer rendering
	if (camAltKM < waterLevelKM) {
		outColor = vec4(currentRadiance, 1.0);
		return;
	}

	bool  hasSurface = albedo.a >= 0.01;
	float surfaceDistKM = hasSurface ? (length(relPos) / 1000.0) : 1e9;

	vec2 clipCoord = inUV * 2.0 - 1.0;
	vec4 viewRay4 = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
	vec3 viewDir = viewRay4.xyz / viewRay4.w;
	vec3 worldRay = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

	float planetRadius = FAKE_PLANET_RADIUS / 1000.0;
	float r = planetRadius + camAltKM;
	r = max(planetRadius + 0.001, r);

	float cloudAlt = 5.0;
	float cloudRadius = planetRadius + cloudAlt;

	float b = 2.0 * r * worldRay.y;
	float c = (r * r) - (cloudRadius * cloudRadius);
	float det = (b * b) - (4.0 * c);

	if (det > 0.0) {
		float sqrtDet = sqrt(det);
		float t1 = (-b - sqrtDet) * 0.5;
		float t2 = (-b + sqrtDet) * 0.5;

		vec3 sunDir = normalize(push.sunDir.xyz);
		vec3 sunRadiance = push.sunRadianceAndSkyExp.xyz;
		vec3 skyRadiance = sampleSkyView(push.skyViewIndex, worldRay);

		// If camera is above the cloud layer (r > cloudRadius), check far intersection (t2) first, then near (t1).
		// If camera is below or inside, only t2 is forward (t1 <= 0).
		if (t2 > 0.0 && t2 < surfaceDistKM) {
			// If camera is above cloud layer, t2 represents the far limb intersection
			if (r > cloudRadius) {
				currentRadiance += calculateCloudColor(t2, worldRay, worldScale, planetRadius, cloudAlt, sunDir, sunRadiance, skyRadiance);
			} else if (t1 <= 0.0) {
				// Camera below cloud layer, t2 is the single forward intersection looking up
				currentRadiance += calculateCloudColor(t2, worldRay, worldScale, planetRadius, cloudAlt, sunDir, sunRadiance, skyRadiance);
			}
		}

		if (t1 > 0.0 && t1 < surfaceDistKM) {
			// Near intersection (top side when camera is above, or entry point)
			currentRadiance += calculateCloudColor(t1, worldRay, worldScale, planetRadius, cloudAlt, sunDir, sunRadiance, skyRadiance);
		}
	}

	outColor = vec4(currentRadiance, 1.0);

	// // Sample from the 3 volume texture cascades (minimally populated sample)
	// vec4 cascade0Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.x, vec3(vUv, 0.5));
	// vec4 cascade1Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, vec3(vUv, 0.5));
	// vec4 cascade2Sample = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, vec3(vUv, 0.5));

	// vec4 cloudVal = cascade0Sample * 0.5 + cascade1Sample * 0.3 + cascade2Sample * 0.2;
	// outColor = cloudVal;
}
