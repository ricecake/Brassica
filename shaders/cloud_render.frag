#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "lighting.glsl"
#include "atmosphere/common.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/cloud_shadow.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "common.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outCloudColor; // rgb = lightEnergy, a = cloudTransmittance
layout(location = 1) out float outDepth;     // ray depth / distance in meters
layout(location = 2) out vec2 outVelocity;   // motion vector (screen space UV delta)

layout(push_constant) uniform CloudRenderPushConstants {
	vec4  sunDir;
	vec4  sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	uvec4 cascadeSampledIdx; // x, y, z = sampled bindless indices for cascades 0, 1, 2
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  transmittanceIndex;
	uint  skyViewIndex;
	uint  weatherBiomeIndex;
	uint  cloudShadowIndex;
	float cirrusOpacity;
} push;

vec4 sampleCloudShadow(vec3 p) {
	return sampleCloudShadow(push.cloudShadowIndex, p, push.sunDir.xyz, uCameraPosition.xyz, 320000.0);
}

vec3 getTransmittance(float r, float mu) {
	vec2 uv = transmittanceToUV(r, mu);
	return SAMPLE_LINEAR(push.transmittanceIndex, uv).rgb;
}

// Assumes footprints: C0 (20km), C1 (80km), C2 (320km)
float sampleCloudCascades(vec3 worldPos, float distFromCam) {
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;

	float uv_y = (adjustedY - 5000.0) / 15000.0;

	if (uv_y < 0.0 || uv_y > 1.0) return 0.0;

	vec3 uvw0 = vec3(fract(worldPos.x / 20000.0),  uv_y, fract(worldPos.z / 20000.0));
	vec3 uvw1 = vec3(fract(worldPos.x / 80000.0),  uv_y, fract(worldPos.z / 80000.0));
	vec3 uvw2 = vec3(fract(worldPos.x / 320000.0), uv_y, fract(worldPos.z / 320000.0));

	float density = 0.0;

	if (distFromCam < 10000.0) {
		float d0 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.x, uvw0).r;
		if (distFromCam > 8000.0) {
			float d1 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, uvw1).r;
			density = mix(d0, d1, smoothstep(8000.0, 10000.0, distFromCam));
		} else {
			density = d0;
		}
	} else if (distFromCam < 40000.0) {
		float d1 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.y, uvw1).r;
		if (distFromCam > 32000.0) {
			float d2 = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, uvw2).r;
			density = mix(d1, d2, smoothstep(32000.0, 40000.0, distFromCam));
		} else {
			density = d1;
		}
	} else {
		density = SAMPLE_3D_LINEAR(push.cascadeSampledIdx.z, uvw2).r;
	}

	return density;
}

vec3 get_curl(mat3 J) {
    return vec3(
        J[1][2] - J[2][1],
        J[2][0] - J[0][2],
        J[0][1] - J[1][0]
    );
}

// Returns vec4(lightEnergy, cloudTransmittance) and outputs ray hitDepth
vec4 marchClouds(vec3 worldRay, float t_start, float t_end, vec3 sunDir, vec3 sunRadiance, out float hitDepth) {
	float stepSize = 50.0;
	float t = t_start;

	vec2 fragCoord = gl_FragCoord.xy;
	float jitter = InterleavedGradientNoise(fragCoord, int(uFrameIndex));
	t += stepSize * jitter;

	float cloudTransmittance = 1.0;
	vec3 lightEnergy = vec3(0.0);
	float prevDensity = 0.0;

	float accumWeight = 0.0;
	float weightedDepth = 0.0;

	const float cloudPhaseG1 = 0.8;
	const float cloudPhaseG2 = -0.2;
	const float cloudPhaseAlpha = 0.5;
	const float cloudPhaseIsotropic = 0.52;
	const float cloudBeerPowderMix = 0.5;

	while (t < t_end) {
		vec3 p_cloud = uCameraPosition.xyz + worldRay * t;
		float distFromCam = t;

		vec3 planetCenter = vec3(0.0, -FAKE_PLANET_RADIUS, 0.0);
		vec3 toPlanetCenter = p_cloud - planetCenter;
		float r_world = length(toPlanetCenter);
		vec3 surfaceDir = normalize(toPlanetCenter);
		float worldExtent = 2.0 * FAKE_PLANET_RADIUS * 3.14159265359;
		vec2 weatherUV = fract((p_cloud.xz / worldExtent) + 0.5);
		vec4 weatherSample = SAMPLE_LINEAR(push.weatherBiomeIndex, weatherUV);
		float weatherCoverage = weatherSample.r;
		float weatherType = weatherSample.g;
		float weatherMoisture = weatherSample.b;

		if (weatherCoverage > 0.01) {
			float progress = 1.0 - smoothstep(t_start, 1.25 * t_end, t);
			mat3 grad;
			vec3 crNoise = cross_noise_fbm(p_cloud / 75.0, 4, uTime * 0.5, grad);
			vec3 cross_curl = get_curl(grad);
			float baseDensity = weatherCoverage * sampleCloudCascades(p_cloud + 100.0 * progress * crNoise, distFromCam);
			float density = 1.0 * baseDensity * (0.5 + 1.5 * weatherMoisture);

			float dtNoise = dot_noise_fbm((p_cloud + 25.0 * progress * cross_curl) / 750.0, 4, 0.05 * uTime);
			density = remapClamp(baseDensity, abs(dtNoise / 4.0), 1.0, 0.0, 1.0);
			density = 2.0 * smoothstep(0.15, 0.85, density);

			if (density > 0.0) {
				float extinction = max(0.0, density * (0.005 + 0.02 * weatherMoisture));
				float averageDensity = (prevDensity + density) * 0.5;
				float stepDensity = averageDensity * extinction;
				float stepOpticalDepth = stepDensity * stepSize;
				float stepTransmittance = exp(-stepOpticalDepth);

				vec4 cloudShadowData = sampleCloudShadow(p_cloud);
				float opticalDepthToLight = cloudShadowData.b;

				float r_km = r_world / 1000.0;
				float mu = dot(surfaceDir, sunDir);
				vec3 atmosphereTransmittance = getTransmittance(r_km, mu);

				float cosTheta = dot(worldRay, sunDir);
				const int OCTAVES = 3;
				float currentExtinction = 1.0;
				float currentKfwd = cloudPhaseG1;
				float currentKbck = cloudPhaseG2;
				float currentEnergy = 1.0;
				float currentScatChaos = (1.0 - cloudPhaseIsotropic);

				vec3 msScattering = vec3(0.0);

				for(int oct = 0; oct < OCTAVES; oct++) {
					float phase = dualLobeSchlick(cosTheta, currentKfwd, currentKbck, cloudPhaseAlpha);
					phase = mix((1.0 / (4.0 * PI)), phase, currentScatChaos);

					float octaveOpticalDepth = opticalDepthToLight * currentExtinction;
					float octaveStepDensity = stepDensity * currentExtinction;

					float shadowTerm = mix(
						beerPowder(octaveOpticalDepth, octaveStepDensity),
						exp(-octaveOpticalDepth),
						cloudBeerPowderMix
					);

					msScattering += vec3(shadowTerm * phase * currentEnergy);

					currentExtinction *= 0.5;
					currentKfwd *= 0.5;
					currentKbck *= 0.5;
					currentEnergy *= 0.5;
					currentScatChaos *= 0.5;
				}

				const float powderScale = 300.0;
				float powder = 1.0 - 1.0 / (1.0 + (25.0 * stepDensity * powderScale));

				vec3 ambientGlow = sunRadiance * vec3(0.02, 0.03, 0.04) * powder;
				vec3 directLight = sunRadiance * atmosphereTransmittance * msScattering;
				vec3 stepScattering = directLight + ambientGlow;

				vec3 inscatter = stepScattering * (1.0 - stepTransmittance);

				float stepWeight = (1.0 - stepTransmittance) * cloudTransmittance;
				weightedDepth += t * stepWeight;
				accumWeight += stepWeight;

				lightEnergy += inscatter * cloudTransmittance;
				cloudTransmittance *= stepTransmittance;

				if (cloudTransmittance < 0.05) break;
			}
			prevDensity = density;
		}

		t += stepSize;
		stepSize *= 1.02;
	}

	hitDepth = (accumWeight > 0.0001) ? (weightedDepth / accumWeight) : 1e9;
	return vec4(lightEnergy, cloudTransmittance);
}

void main() {
	vec4 albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3 relPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;

	float worldScale = 1.0;
	float waterLevelKM = u_waterLevel / 1000.0;
	float camAltKM = uCameraPosition.y / (1000.0 * worldScale);

	// When submerged underwater, skip cloud layer rendering
	if (camAltKM < waterLevelKM) {
		outCloudColor = vec4(0.0, 0.0, 0.0, 1.0);
		outDepth = 1e9;
		outVelocity = vec2(0.0);
		return;
	}

	bool  hasSurface = albedo.a >= 0.01;
	float surfaceDist = hasSurface ? length(relPos) : 1e9;

	vec2 clipCoord = inUV * 2.0 - 1.0;
	vec4 viewRay4 = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
	vec3 viewDir = viewRay4.xyz / viewRay4.w;
	vec3 worldRay = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);

	float cloudAlt = 5000.0;
	float t_s1, t_e1, t_s2, t_e2;
	bool intersection = intersectCloudShell(uCameraPosition.xyz, worldRay, FAKE_PLANET_RADIUS, cloudAlt, 15000, t_s1, t_e1, t_s2, t_e2);

	vec4 result = vec4(0.0, 0.0, 0.0, 1.0);
	float cloudDepth = 1e9;

	if (intersection) {
		vec3 sunDir = normalize(push.sunDir.xyz);
		vec3 sunRadiance = push.sunRadianceAndSkyExp.xyz;

		// Segment 1 (Near side of the cloud shell)
		if (t_s1 < t_e1 && t_s1 < surfaceDist) {
			float depth1 = 1e9;
			vec4 near = marchClouds(worldRay, t_s1, min(t_e1, surfaceDist), sunDir, sunRadiance, depth1);
			result.rgb += near.rgb;
			result.a *= near.a;
			if (depth1 < 1e8) cloudDepth = depth1;
		}

		// Segment 2 (Far side of the cloud shell)
		if (t_s2 < t_e2 && t_s2 < surfaceDist) {
			float depth2 = 1e9;
			vec4 far = marchClouds(worldRay, t_s2, min(t_e2, surfaceDist), sunDir, sunRadiance, depth2);
			result.rgb += far.rgb * result.a;
			result.a *= far.a;
			if (cloudDepth > 1e8 && depth2 < 1e8) cloudDepth = depth2;
		}
	}

	// Calculate screen space motion vector for cloud reprojection
	vec2 velocity = vec2(0.0);
	if (cloudDepth < 1e8) {
		vec3 cloudWorldPos = uCameraPosition.xyz + worldRay * cloudDepth;
		vec4 prevClip = uPreviousViewProjMatrix * vec4(cloudWorldPos, 1.0);
		if (prevClip.w > 0.001) {
			vec2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
			velocity = inUV - prevUV;
		}
	}

	outCloudColor = result;
	outDepth = cloudDepth;
	outVelocity = velocity;
}
