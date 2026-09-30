#version 460
#include "bindless.glsl"
#include "lighting.glsl"
#include "common.glsl"
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
} push;

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

void main() {
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

	vec3 cirrusColor = vec3(0.0);

	if (det > 0.0) {
		float sqrtDet = sqrt(det);
		float t1 = (-b - sqrtDet) * 0.5;
		float t2 = (-b + sqrtDet) * 0.5;

		float t_cirrus = -1.0;
		if (t1 > 0.0) {
			t_cirrus = t1;
		} else if (t2 > 0.0) {
			t_cirrus = t2;
		}

		// Draw cirrus clouds if intersection exists and is in front of opaque scene geometry
		if (t_cirrus > 0.0 && t_cirrus < surfaceDistKM) {
			vec3 p_cirrus = uCameraPosition.xyz + worldRay * (t_cirrus * 1000.0 * worldScale);

			vec3 advect = vec3(1.0, 0.0, 1.0) * uTime * 0.5;
			vec2 uv_cirrus = (p_cirrus.xz + advect.xz) * (0.00005 / worldScale);

			float n = (fbm_astral(vec3(uv_cirrus * 2.0, uTime * 0.01)) + 1.0) * 0.5;
			float n2 = (fbm_astral(vec3(uv_cirrus * 5.0, uTime * 0.02 + 10.0)) + 1.0) * 0.5;
			float noise = smoothstep(0.3, 0.8, n * n2);

			vec3  sunDir = normalize(push.sunDir.xyz);
			vec3  sunRadiance = push.sunRadianceAndSkyExp.xyz;
			vec3  skyRadiance = sampleSkyView(push.skyViewIndex, worldRay);
			vec3  T_cirrus = max(getTransmittance(planetRadius + cirrusAlt, sunDir.y), vec3(0.001));
			float cirrusPhase = mix(0.2, 1.0, pow(max(0.0, dot(worldRay, sunDir)), 3.0));

			float opacity = push.cirrusOpacity > 0.0 ? push.cirrusOpacity : 0.0125;
			vec3  cirrusLighting = (T_cirrus * sunRadiance * cirrusPhase * 5.0) + (skyRadiance * 0.5);
			cirrusColor = cirrusLighting * noise * opacity * 15.0;

			float opticalDepthFade = exp(-t_cirrus * 0.0025);
			cirrusColor *= opticalDepthFade;

			// Proximity fade: As the camera gets closer to the layer, the area near the camera becomes transparent
			// to avoid directly seeing how thin the clouds are when passing through.
			float proximityFade = smoothstep(0.1, 3.0, t_cirrus);
			cirrusColor *= proximityFade;
		}
	}

	outColor = vec4(currentRadiance + cirrusColor, 1.0);
}
