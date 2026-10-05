#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "lighting.glsl"
#include "atmosphere/common.glsl"
#include "helpers/octahedral.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "common.glsl"

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
/*
struct CloudProperties {
	float altitude;
	float thickness;
	float densityBase;
	float coverage;
	float worldScale;
};

struct CloudWeather {
	vec3 p;
	float coverage;
	float density;
	float heightMap;  // Altitude variety
	float thickness;  // Thickness variety
	float ridge;      // Ridge noise weighted by Worley F1 distance
	float ecentricity;
	float curve;
	float centerDist;
	float baseFloor;
	float baseCeiling;
	float height;
	float moisture;
	float humidity;
};

vec3 EvaluateStepLighting(
	vec3 p,
	vec3 rayDir,
	float jitter,
	CloudWeather weather,
	CloudProperties props,
	vec3 rgbExtinction,
	vec3 stepDensity,
	vec3 stepOpticalDepth,
	float h_norm,
	vec3 sampleAdvection,
	float sampleAo,
	vec3 sampleAlbedo,
	inout vec3 emissiveGlow,
	vec3 skySH,
	vec3 horizonSH,
	vec3 groundSH,
	vec3 primaryLightDir,
	float R_ceiling,
	float R_floor,
	vec3 earthCenter,
	float currentLOD
) {
	vec3 stepScattering = vec3(0.0);
	float edgeDistance = 150.0;//getDistanceToCloudEdge(weather.coverage, h_norm, weather.heightMap, weather.thickness);
	vec3 localAmbientVisibility = exp(-edgeDistance * stepDensity * 0.05);

	for (int j = 0; j < min(2, num_lights); j++) {
		if (lights[j].type != LIGHT_TYPE_DIRECTIONAL) continue;
		if (lights[j].intensity <= 0.0) continue;

		if ((lights[j].flags & LIGHT_FLAG_CASTS_SHADOW) == 0 ||
			(lights[j].flags & LIGHT_FLAG_VOLUMETRIC_SHADOW) == 0) continue;

		vec3 L;
		float attenuation;
		calculateLightContribution(j, p, L, attenuation);

		if (attenuation * lights[j].intensity < 0.001) continue;

		float traceDist = 0.0;
		vec3 lightTransmittance = vec3(1.0);

		float r_world = length(p - earthCenter);
		float r_km = r_world / (1000.0 * worldScale);
		float mu = dot((p - earthCenter) / r_world, L);

		float st_start = 1e10;
		float st_end = -1e10;
		intersectCloudShell(p, L, worldScale, st_start, st_end);
		traceDist = min(st_end - st_start, props.thickness * 1.5 * worldScale);

		vec2 transUV = getTransmittanceUV(r_km, mu);
		lightTransmittance = texture(u_transmittanceLUT, transUV).rgb;

		vec3 opticalDepthToLight = vec3(0.0);
		if (u_useCloudShadowMap && j == 0) {
			vec4 lightSpacePos = u_cloudShadowMatrix * vec4(p, 1.0);
			vec2 shadowUV = lightSpacePos.xy * 0.5 + 0.5;
			float accumulatedDensity = sampleDeepOpacityMap(shadowUV, h_norm, 0.0);
			opticalDepthToLight = accumulatedDensity * (rgbExtinction * cloudShadowOpticalDepthMultiplier);
		}
		else if (traceDist > 0.01 * worldScale) {
			float current_t = 20.0 + (100.0 * jitter);
			float current_dt = 100.0;
			for (int i = 0; i < 5; i++) {
				if (current_t > traceDist) break;
				vec3 sp_c = p + L * current_t;
				CloudWeather shadow_weather_c = computeCloudWeather(sp_c, props, currentLOD);
				CloudDensityResult shadowRes_c = calculateCloudDensity(sp_c, shadow_weather_c, props, time, 0.0, true);
				opticalDepthToLight += max(vec3(0.00), shadowRes_c.density) * current_dt * (rgbExtinction * cloudShadowOpticalDepthMultiplier);
				emissiveGlow += shadowRes_c.emissivity * exp(-opticalDepthToLight);
				current_t += current_dt;
				current_dt *= 2.0;
			}
		}

		float cosTheta = dot(rayDir, L);

		const int OCTAVES = 3;
		const float extinctionMult = 0.5;
		const float phaseWidenMult = 0.5;
		const float energyAttenuation = 0.5;
		const float msCatChaos = 0.5;

		float kFwd = cloudPhaseG1;
		float kBck = cloudPhaseG2;
		float phaseWeight = cloudPhaseAlpha;

		vec3 msScattering = vec3(0.0);
		float currentExtinction = 1.0;
		float currentKfwd = kFwd;
		float currentKbck = kBck;
		float currentEnergy = 1.0;
		float currentScatChaos = (1.0 - cloudPhaseIsotropic);

		for(int oct = 0; oct < OCTAVES; oct++) {
			float phase = dualLobeSchlick(cosTheta, currentKfwd, currentKbck, phaseWeight);
			phase = mix((1.0 / (4.0 * PI)), phase, currentScatChaos);

			vec3 octaveOpticalDepth = opticalDepthToLight * currentExtinction;
			vec3 octaveStepDensity = stepDensity * currentExtinction;

			vec3 shadowTerm = mix(
				beerPowder(octaveOpticalDepth, octaveStepDensity),
				exp(-octaveOpticalDepth),
				cloudBeerPowderMix
			);

			msScattering += shadowTerm * phase * currentEnergy;

			currentExtinction *= extinctionMult;
			currentKfwd *= phaseWidenMult;
			currentKbck *= phaseWidenMult;
			currentEnergy *= energyAttenuation;
			currentScatChaos *= msCatChaos;
		}

		float lightScale = (j == 0 ? cloudSunLightScale : cloudMoonLightScale);
		stepScattering += lightTransmittance * lights[j].color.rgb * msScattering *
			lights[j].intensity * attenuation * lightScale;
	}

	uint cloud_cluster_index = getClusterIndex(p);
	Cluster cloud_cluster = clusters[cloud_cluster_index];
	Cluster global_cloud_cluster = clusters[3456];

	// CalculateStochasticClusterLight(
	// 	p,
	// 	rayDir,
	// 	jitter,
	// 	weather,
	// 	props,
	// 	rgbExtinction,
	// 	stepOpticalDepth,
	// 	global_cloud_cluster,
	// 	currentLOD,
	// 	stepScattering,
	// 	emissiveGlow
	// );

	// CalculateStochasticClusterLight(
	// 	p,
	// 	rayDir,
	// 	jitter,
	// 	weather,
	// 	props,
	// 	rgbExtinction,
	// 	stepOpticalDepth,
	// 	cloud_cluster,
	// 	currentLOD,
	// 	stepScattering,
	// 	emissiveGlow
	// );

	const float ambientExtinction = 1.0;
	const float powderScale = 300.0;
	vec3 powder = vec3(1.0) - 1.0/(1.0+(25*(stepDensity-0.00) * powderScale));
	float sunHeight = max(primaryLightDir.y, 0.0);
	float ambientScale = mix(0.3, 1.0, smoothstep(0.0, 0.3, sunHeight));

	vec3 ambientTop    = mix(skySH, horizonSH, 0.4) * ambientScale;
	vec3 ambientBottom = groundSH * ambientScale;

	vec3 upwardOD = stepDensity * h_norm * weather.thickness;
	vec3 downwardOD = stepDensity * (1.0 - h_norm) * weather.thickness;
	vec3 groundAttenuation = 1.0 / (1.0 + upwardOD);
	vec3 skyAttenuation = 1.0 / (1.0 + downwardOD);

	vec3 finalAmbient = ((ambientTop * skyAttenuation) + (ambientBottom * groundAttenuation)) * localAmbientVisibility;

	if (lightningPulse > 0.001) {
		float wavePos = clamp(1.0 - lightningPulse, 0.0, 1.0);
		float midHeightEnvelope = exp(-pow((h_norm - 0.5) * 4.0, 2.0));
		float temporalPhase = fract(time * 0.7 + weather.ridge * 3.0);
		float stagger = sin(temporalPhase * 6.28318) * 0.5 + 0.5;

		float ridgeDist = abs(weather.ridge - wavePos * 0.85);
		float waveFront = exp(-ridgeDist * ridgeDist * 16.0);
		float coreGlow = exp(-weather.ridge * 4.0 / max(0.1, lightningPulse));
		float propagationGlow = (coreGlow * 0.7 + waveFront * 1.3) * lightningPulse * midHeightEnvelope * (stagger);

		emissiveGlow += lightningColor * propagationGlow;
	}

	vec3 S = (stepScattering + (finalAmbient * powder)) * (uCloudAlbedo * sampleAlbedo) + emissiveGlow;

	return S;
}
*/

// Assumes footprints: C0 (20km), C1 (80km), C2 (320km)
float sampleCloudCascades(vec3 worldPos, float distFromCam) {
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;

	float uv_y = (adjustedY - 5000.0) / 15000.0;

	// Fixes the vertical streaks (Screenshot 3) by preventing Y-axis wrapping
	if (uv_y < 0.0 || uv_y > 1.0) return 0.0;

	// Fixes the axis-aligned boundaries (Screenshot 1 & 2) by forcing toroidal XZ wrapping
	vec3 uvw0 = vec3(fract(worldPos.x / 20000.0),  uv_y, fract(worldPos.z / 20000.0));
	vec3 uvw1 = vec3(fract(worldPos.x / 80000.0),  uv_y, fract(worldPos.z / 80000.0));
	vec3 uvw2 = vec3(fract(worldPos.x / 320000.0), uv_y, fract(worldPos.z / 320000.0));

	float density = 0.0;

	// Blend radii must match the physical bounds (half the total width)
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

// Change signature to return vec4
vec4 marchClouds(vec3 worldRay, float t_start, float t_end, vec3 sunDir, vec3 sunRadiance) {
	float stepSize = 50.0;
	float t = t_start;

	vec2 fragCoord = gl_FragCoord.xy;
	float jitter = InterleavedGradientNoise(fragCoord, int(uFrameIndex));
	t += stepSize * jitter;

	float cloudTransmittance = 1.0;
	vec3 lightEnergy = vec3(0.0);
	float prevDensity = 0.0;

	const float cloudPhaseG1 = 0.8;
	const float cloudPhaseG2 = -0.2;
	const float cloudPhaseAlpha = 0.05;
	const float cloudPhaseIsotropic = 0.2;
	const float cloudBeerPowderMix = 0.5;

	while (t < t_end) {
		vec3 p_cloud = uCameraPosition.xyz + worldRay * t;
		float distFromCam = t;

		vec3 planetCenter = vec3(0.0, -FAKE_PLANET_RADIUS, 0.0);
		vec3 surfaceDir = normalize(p_cloud - planetCenter);
		vec2 weatherUV = directionToOctahedralUV(surfaceDir);
		float weatherDensity = SAMPLE_LINEAR(push.weatherBiomeIndex, weatherUV).g;

		if (weatherDensity > 0.01) {
			float density = 10*sampleCloudCascades(p_cloud, distFromCam) * weatherDensity;

			if (density > 0.0) {
				float extinction = max(0.0, density * 0.25);
				float averageDensity = (prevDensity + density) * 0.5;
				float stepDensity = averageDensity * extinction;
				float stepOpticalDepth = stepDensity * stepSize;
				float stepTransmittance = exp(-stepOpticalDepth);

				float opticalDepthToLight = 0.0;
				float light_t = stepSize;
				float lightStepSize = stepSize * 2.0;

				for (int i = 0; i < 3; i++) {
					vec3 sp_c = p_cloud + sunDir * light_t;
					float shadow_dist = length(sp_c - uCameraPosition.xyz);
					float shadow_density = sampleCloudCascades(sp_c, shadow_dist) * weatherDensity;

					opticalDepthToLight += max(0.0, shadow_density * 0.0125) * lightStepSize;
					light_t += lightStepSize;
					lightStepSize *= 2.0;
				}

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

					currentExtinction *= 0.25;
					currentKfwd *= 0.5;
					currentKbck *= 0.5;
					currentEnergy *= 0.5;
					currentScatChaos *= 0.5;
				}

				const float powderScale = 300.0;
				float powder = 1.0 - 1.0 / (1.0 + (25.0 * stepDensity * powderScale));

				// Tie ambient glow loosely to sunRadiance so they scale together under HDR
				vec3 ambientGlow = sunRadiance * vec3(0.02, 0.03, 0.04) * powder;

				vec3 stepScattering = (sunRadiance * msScattering) + ambientGlow;

				// FIX: Removed the `/ max(stepDensity, 0.0001)` that was blowing up the light
				vec3 inscatter = stepScattering * (1.0 - stepTransmittance);

				lightEnergy += inscatter * cloudTransmittance;
				cloudTransmittance *= stepTransmittance;

				if (cloudTransmittance < 0.05) break;
			}
			prevDensity = density;
		}

		t += stepSize;
		stepSize *= 1.02;
	}

	// Return energy and the remaining transmittance
	return vec4(lightEnergy, cloudTransmittance);
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
		vec4 result = vec4(0.0, 0.0, 0.0, 1.0);

		// Segment 1 (Near side of the cloud shell)
		if (t_s1 < t_e1 && t_s1 < surfaceDistKM) {
			vec4 near = marchClouds(worldRay, t_s1, min(t_e1, surfaceDistKM), sunDir, sunRadiance);
			result.rgb += near.rgb;
			result.a *= near.a;
		}

		// Segment 2 (Far side of the cloud shell, if ray pierces entirely through space)
		if (t_s2 < t_e2 && t_s2 < surfaceDistKM) {
			vec4 far = marchClouds(worldRay, t_s2, min(t_e2, surfaceDistKM), sunDir, sunRadiance);
			result.rgb += far.rgb * result.a;
			result.a *= far.a;
		}

		currentRadiance = currentRadiance * result.a + result.rgb;
	}
	outColor = vec4(currentRadiance, 1.0);
}
