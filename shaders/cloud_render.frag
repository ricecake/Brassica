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

// Assumes footprints: C0 (20km), C1 (80km), C2 (320km)
float sampleCloudCascades(vec3 worldPos, float distFromCam) {
// 1. Curved Earth Dropoff
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;

	// 2. Exact UV Y-axis mapping matching the compute shader (5000m base, 15000m span)
	float uv_y = (adjustedY - 5000.0) / 15000.0;

	// 3. Absolute XZ mapping (hardware sampler VK_SAMPLER_ADDRESS_MODE_REPEAT handles wrapping)
	vec3 uvw0 = vec3(worldPos.x / 20000.0,  uv_y, worldPos.z / 20000.0);
	vec3 uvw1 = vec3(worldPos.x / 80000.0,  uv_y, worldPos.z / 80000.0);
	vec3 uvw2 = vec3(worldPos.x / 320000.0, uv_y, worldPos.z / 320000.0);
	float density = 0.0;

	if (distFromCam < 20000.0) {
		// return 0.0;
		float d0 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.x, uvw0).r;
		if (distFromCam > 16000.0) {
			// Blend zone: 16km to 20km
			float d1 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, uvw1).r;
			float blend = smoothstep(16000.0, 20000.0, distFromCam);
			density = mix(d0, d1, blend);
		} else {
			density = d0;
		}
	} else if (distFromCam < 80000.0) {
		return 0.0;
		float d1 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, uvw1).r;
		if (distFromCam > 64000.0) {
			// Blend zone: 64km to 80km
			float d2 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, uvw2).r;
			float blend = smoothstep(64000.0, 80000.0, distFromCam);
			density = mix(d1, d2, blend);
		} else {
			density = d1;
		}
	} else {
		// return 0.0;
		// Fallback to lowest detail cascade
		density = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, uvw2).r;
	}

	return density;
}

float rangeInterpolate(float dist, float res, float f0, float f1, float f2) {
	if (dist <= f0) return f0;
	if (dist >= f2) return f2;
	if (dist >= f1) {
		return f1 + smoothstep(f2 - 0.25*(f2-f1), f2, dist);
	}
	if (dist >= f0) {
		return f0 + smoothstep(f1 - 0.25*(f1-f0), f1, dist);
	}
}

vec3 marchClouds(vec3 worldRay, float t_start, float t_end, vec3 sunDir, vec3 sunRadiance) {
	// 1. Initial Setup and Jitter
	float stepSize = 50.0; // Base step size (meters)
	float t = t_start;

	// Jitter the start position using Interleaved Gradient Noise or Bayer matrix
	// to hide the discrete steps and prevent view-movement banding.
	vec2 fragCoord = gl_FragCoord.xy;
	float jitter = fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));
	t += stepSize * jitter;

	vec4 accumulatedColor = vec4(0.0); // rgb = color, a = accumulated alpha
	float transmittance = 1.0;

	// 2. Integration Loop
	while (t < t_end) {
		vec3 p_cloud = uCameraPosition.xyz + worldRay * t;
		float distFromCam = t;

		// Space skipping using the 2D weather map
		vec3 planetCenter = vec3(0.0, -FAKE_PLANET_RADIUS, 0.0);
		vec3 surfaceDir = normalize(p_cloud - planetCenter);
		vec2 weatherUV = directionToOctahedralUV(surfaceDir);
		float weatherDensity = SAMPLE_LINEAR(push.weatherBiomeIndex, weatherUV).g;

		if (weatherDensity > 0.01) {
			float density = sampleCloudCascades(p_cloud, distFromCam) * weatherDensity;

			if (density > 0.0) {
				// Optical depth for this single step
				float extinction = max(0.0, density * 0.025);
				float stepTransmittance = exp(-extinction * stepSize);

				// Simple lighting formulation for the step
				float cloudPhase = mix(0.2, 1.0, pow(max(0.0, dot(worldRay, sunDir)), 3.0));
				vec3 stepLight = sunRadiance * cloudPhase * density * 0.01; // Expand with multiple-scattering later

				// Accumulate front-to-back
				vec3 inscatter = stepLight * (1.0 - stepTransmittance) / max(extinction, 0.0001);
				accumulatedColor.rgb += inscatter * transmittance;

				transmittance *= stepTransmittance;
				accumulatedColor.a = 1.0 - transmittance;

				// Early exit if completely opaque
				if (transmittance < 0.01) break;
			}
		}

		// 3. Variable Step Sizing (LOD Stepping)
		t += stepSize;
		// Increase step size geometrically to span horizon distances
		stepSize *= 1.02;
	}

	return accumulatedColor.rgb;
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
	float surfaceDistKM = hasSurface ? (length(relPos)) : 1e9;

	vec2 clipCoord = inUV * 2.0 - 1.0;
	vec4 viewRay4 = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
	vec3 viewDir = viewRay4.xyz / viewRay4.w;
	vec3 worldRay = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

	float cloudAlt = 5000.0;
	float t_s1, t_e1, t_s2, t_e2;
	bool intersection = intersectCloudShell(uCameraPosition.xyz, worldRay, FAKE_PLANET_RADIUS, cloudAlt, 15000, t_s1, t_e1, t_s2, t_e2);
	if (intersection) {
		vec3 sunDir = normalize(push.sunDir.xyz);
		vec3 sunRadiance = push.sunRadianceAndSkyExp.xyz;
		vec3 accumulated = vec3(0.0);

		// Segment 1 (Near side of the cloud shell)
		if (t_s1 < t_e1 && t_s1 < surfaceDistKM) {
			accumulated += marchClouds(worldRay, t_s1, min(t_e1, surfaceDistKM), sunDir, sunRadiance);
		}

		// Segment 2 (Far side of the cloud shell, if ray pierces entirely through space)
		if (t_s2 < t_e2 && t_s2 < surfaceDistKM) {
			accumulated += marchClouds(worldRay, t_s2, min(t_e2, surfaceDistKM), sunDir, sunRadiance);
		}

		currentRadiance = mix(currentRadiance, accumulated, smoothstep(0.1, 1.95, accumulated));
	}
	outColor = vec4(currentRadiance, 1.0);
}
