#version 460
#include "bindless.glsl"
#include "lighting.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform SkyPushConstants {
	vec4 sunDirAndAureole;      // xyz = sunDir, w = sunAureole
	vec4 moonDirAndCirrus;     // xyz = moonDir, w = cirrus
	vec4 sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	float worldScale;
	uint  skyViewIndex;
	uint  transmittanceIndex;
	float padding;
} push;

#include "common.glsl"
#include "helpers/astral.glsl"

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

void main() {
	vec2 clipCoord = inUV * 2.0 - 1.0;
	vec4 viewRay = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
	vec3 worldRay = normalize((uInvViewMatrix * vec4(viewRay.xy, -1.0, 0.0)).xyz);

	float worldScale = max(0.001, push.worldScale);
	float r = kEarthRadius + max(0.0, uCameraPosition.y / (1000.0 * worldScale));

	// 1. Sky Radiance from SkyViewLUT
	vec3 skyRadiance = sampleSkyView(push.skyViewIndex, worldRay);

	// 2. Sun Disc with Aureole
	vec3  sunDir = normalize(push.sunDirAndAureole.xyz);
	vec3  sunRadiance = push.sunRadianceAndSkyExp.xyz;
	float sunAureoleStrength = push.sunDirAndAureole.w;

	vec3 sunUp = vec3(0, 1, 0);
	vec3 sunRight = normalize(cross(sunUp, sunDir));
	if (length(sunRight) < 0.001)
		sunRight = vec3(1, 0, 0);
	sunUp = cross(sunDir, sunRight);

	float rayLocalX = dot(worldRay, sunRight);
	float rayLocalY = dot(worldRay, sunUp);
	float rayLocalZ = dot(worldRay, sunDir);

	float atmosphericRefraction = 1.0 + 0.6 * exp(-max(0.0, sunDir.y * 10.0));
	float flattenedY = rayLocalY * atmosphericRefraction;

	float distSq = rayLocalX * rayLocalX + flattenedY * flattenedY;
	float distToSun = sqrt(max(0.0, distSq));

	float sunAngularRadius = 0.02; // ~1 degree
	float aureole = exp(-distToSun * 40.0) * sunAureoleStrength * 3.5;

	float sunMask = 1.0 - smoothstep(
		(sunAngularRadius - 0.001) * (sunAngularRadius - 0.001),
		sunAngularRadius * sunAngularRadius,
		distSq
	);
	sunMask += aureole;
	sunMask = (sunMask * (1.0 + sunMask * 0.05)) / (1.0 + sunMask * 0.06);
	sunMask *= step(0.0, rayLocalZ);

	vec3 sunTransmittance = max(getTransmittance(r, sunDir.y), vec3(0.001));
	vec3 sunDisc = sunRadiance * sunMask * sunTransmittance * smoothstep(-0.01, 0.01, sunDir.y);

	// 3. Stars and Nebula
	float skyBrightness = max(max(skyRadiance.r, skyRadiance.g), skyRadiance.b);
	float starVisibility = 1.0 - smoothstep(0.05, 0.5, skyBrightness);
	vec3 spaceBackground = vec3(0.0);

	if (starVisibility > 0.0) {
		vec3 skyTransmittance = getTransmittance(r, worldRay.y);
		if (any(greaterThan(skyTransmittance, vec3(0.0)))) {
			vec3 stars = computeStars(worldRay, uTime);
			vec3 nebula = computeNebula(worldRay, uTime);
			spaceBackground = (stars + nebula) * skyTransmittance * starVisibility;
		}
	}

	// 4. Moon Disc
	vec3  moonDir = normalize(push.moonDirAndCirrus.xyz);
	vec3  moonRadiance = vec3(0.12, 0.14, 0.18) * push.sunRadianceAndSkyExp.w;
	float moonAngularRadius = 0.018;

	vec3 moonUp = vec3(0, 1, 0);
	vec3 moonRight = normalize(cross(moonUp, moonDir));
	if (length(moonRight) < 0.001)
		moonRight = vec3(1, 0, 0);
	moonUp = cross(moonDir, moonRight);

	float moonLocalX = dot(worldRay, moonRight);
	float moonLocalY = dot(worldRay, moonUp);
	float moonLocalZ = dot(worldRay, moonDir);

	float moonFlattenedY = moonLocalY * (1.0 + 0.6 * exp(-max(0.0, moonDir.y * 10.0)));
	float moonDistSq = moonLocalX * moonLocalX + moonFlattenedY * moonFlattenedY;

	float moonMask = smoothstep(
		moonAngularRadius * moonAngularRadius,
		(moonAngularRadius - 0.001) * (moonAngularRadius - 0.001),
		moonDistSq
	);
	moonMask *= step(0.99, moonLocalZ);

	float r2 = dot(
		vec2(moonLocalX, moonFlattenedY) / moonAngularRadius,
		vec2(moonLocalX, moonFlattenedY) / moonAngularRadius
	);
	float z = sqrt(max(0.0, 1.0 - r2));
	vec3  moonNormal = normalize(moonLocalX * moonRight + moonFlattenedY * moonUp + z * moonDir);
	float moonIllum = max(0.0, dot(moonNormal, sunDir)) * 5.8;
	float phasedMask = moonMask * mix(0.02, 1.0, moonIllum);

	vec3 moonTransmittance = max(getTransmittance(r, moonDir.y), vec3(0.001));
	vec3 moonDisc = moonRadiance * phasedMask * moonTransmittance * smoothstep(-0.01, 0.01, moonDir.y);

	// 5. Scaffolding Hooks: Volumetric Lighting & Raymarched Clouds
	vec4 cloudResult = evaluateRaymarchedClouds(uCameraPosition.xyz / 1000.0, worldRay, 100.0, sunDir, sunRadiance);
	vec3 skyWithClouds = mix(skyRadiance, cloudResult.rgb, cloudResult.a);

	vec3 finalColor = skyWithClouds + sunDisc + moonDisc + spaceBackground;

	// Underwater fog used to be applied here too, redundantly and inconsistently with
	// deferred.frag's own (broken) copy. Now handled uniformly for every pixel -- this
	// background included, once DeferredNode copies it into HdrColor for empty G-buffer pixels --
	// by AtmosphereCompositeNode, which runs after both nodes.
	outColor = vec4(finalColor, 1.0);
}
