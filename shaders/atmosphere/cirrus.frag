#version 460
#include "bindless.glsl"
#include "lighting.glsl"
#include "common.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/astral.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CirrusPushConstants {
	vec4  sunDir;
	vec4  sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	float worldScale;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  hdrColorIndex;
	uint  transmittanceIndex;
	uint  skyViewIndex;
	float cirrusAlt;
	float cirrusOpacity;
	uint  weatherBiomeIndex;
} push;

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

vec3 calculateCirrusColor(
	float t_cirrus,
	vec3  worldRay,
	float worldScale,
	float planetRadius,
	float cirrusAlt,
	vec3  sunDir,
	vec3  sunRadiance,
	vec3  skyRadiance
) {
	vec3 p_cirrus = uCameraPosition.xyz + worldRay * (t_cirrus * 1000.0 * worldScale);

	vec3 planetCenter = vec3(0.0, -FAKE_PLANET_RADIUS, 0.0);
	vec3 surfaceDir = normalize(p_cirrus - planetCenter);
	vec2 weatherUV = directionToOctahedralUV(surfaceDir);

	float weatherCloudDensity = 0.5;
	float rainfall = 0.0;

	if (push.weatherBiomeIndex > 0u) {
		vec4 weatherSample = SAMPLE_LINEAR(push.weatherBiomeIndex, weatherUV);
		weatherCloudDensity = clamp(weatherSample.g, 0.0, 1.0);
		rainfall = clamp(weatherSample.b, 0.0, 1.0);
	}

	// Advect cirrus clouds with 2D/3D atmospheric circulation wind map
	float latitude = abs(surfaceDir.y);
	float wEast = -0.012 * sin(3.0 * latitude * PI);
	float wNorth = -0.004 * sin(2.0 * latitude * PI) * (surfaceDir.y >= 0.0 ? 1.0 : -1.0);

	vec3 up = abs(surfaceDir.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	vec3 eastTang = normalize(cross(up, surfaceDir));
	vec3 northTang = normalize(cross(surfaceDir, eastTang));
	vec3 windDir3D = eastTang * wEast + northTang * wNorth;

	vec3 advect3D = windDir3D * (uTime * 50000.0) + vec3(1.0, 0.0, 1.0) * (uTime * 0.2);
	vec2 uv_cirrus = (p_cirrus.xz + advect3D.xz) * (0.00005 / worldScale);

	float n = (fbm_astral(vec3(uv_cirrus * 2.0, uTime * 0.01)) + 1.0) * 0.5;
	float n2 = (fbm_astral(vec3(uv_cirrus * 5.0, uTime * 0.02 + 10.0)) + 1.0) * 0.5;

	// Modulate noise threshold and coverage by weather system cloud density
	float effectiveDensity = mix(0.4, 1.0, weatherCloudDensity);
	float noiseCutoff = mix(0.55, 0.15, effectiveDensity);
	float noise = smoothstep(noiseCutoff, noiseCutoff + 0.35, n * n2);

	vec3  T_cirrus = max(getTransmittance(planetRadius + cirrusAlt, sunDir.y), vec3(0.001));
	float cirrusPhase = mix(0.2, 1.0, pow(max(0.0, dot(worldRay, sunDir)), 3.0));

	float baseOpacity = push.cirrusOpacity > 0.0 ? push.cirrusOpacity : 0.0125;
	float opacity = baseOpacity * (0.6 + 1.4 * effectiveDensity);

	vec3  cirrusLighting = (T_cirrus * sunRadiance * cirrusPhase * 5.0) + (skyRadiance * 0.5);

	// Darken clouds where there is rainfall
	float rainDarkening = mix(1.0, 0.35, rainfall);
	cirrusLighting *= rainDarkening;

	vec3  cirrusColor = cirrusLighting * noise * opacity * 15.0;

	float opticalDepthFade = exp(-t_cirrus * 0.0025);
	cirrusColor *= opticalDepthFade;

	// Proximity fade: As the camera gets closer to the layer, the area near the camera becomes transparent
	// to avoid directly seeing how thin the clouds are when passing through.
	float proximityFade = smoothstep(0.1, 3.0, t_cirrus);
	cirrusColor *= proximityFade;

	return cirrusColor;
}

void main() {
	return;
	vec3 currentRadiance = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;
	vec4 albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3 relPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;

	float worldScale = max(0.001, push.worldScale);
	float waterLevelKM = u_waterLevel / 1000.0;
	float camAltKM = uCameraPosition.y / (1000.0 * worldScale);

	// When submerged underwater, skip cirrus cloud layer rendering
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

	float cirrusAlt = push.cirrusAlt > 0.0 ? push.cirrusAlt : 10.0;
	float cloudRadius = planetRadius + cirrusAlt;

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
				currentRadiance += calculateCirrusColor(t2, worldRay, worldScale, planetRadius, cirrusAlt, sunDir, sunRadiance, skyRadiance);
			} else if (t1 <= 0.0) {
				// Camera below cloud layer, t2 is the single forward intersection looking up
				currentRadiance += calculateCirrusColor(t2, worldRay, worldScale, planetRadius, cirrusAlt, sunDir, sunRadiance, skyRadiance);
			}
		}

		if (t1 > 0.0 && t1 < surfaceDistKM) {
			// Near intersection (top side when camera is above, or entry point)
			currentRadiance += calculateCirrusColor(t1, worldRay, worldScale, planetRadius, cirrusAlt, sunDir, sunRadiance, skyRadiance);
		}
	}

	outColor = vec4(currentRadiance, 1.0);
}
