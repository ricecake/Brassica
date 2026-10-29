#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "lighting.glsl"
#include "atmosphere/common.glsl"
#include "helpers/octahedral.glsl"
#include "helpers/cloud_shadow.glsl"
#include "helpers/ambient.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "common.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CloudRenderPushConstants {
	vec4  sunDir;
	vec4  sunRadianceAndSkyExp; // xyz = sunRadiance, w = skyExposure
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  hdrColorIndex;
	uint  transmittanceIndex;
	uint  skyViewIndex;
	uint  weatherBiomeIndex;
	uint  cloudShadowIndex;
	uint  clipmapIndex;
	uint  volumeIndex; // sampled bindless index for CloudDetailVolume
} push;

// Wind-driven advection speed for cloud sampling (m/s, XZ plane). Independent of the global
// Hadley/Ferrel prevailing-wind model (helpers/wind.glsl) by design -- this is a dedicated
// artist control for cloud drift, matching boidish's own separate cloudFlowSpeed/Direction
// knobs rather than tying cloud motion to the same large-scale circulation that drives the CA
// weather sim. The *10.0 and the shear formula are boidish's own
// getCloudWindSpeed/getCloud3DNoiseAdvectionSpeed constants verbatim (not re-derived or
// re-scaled -- "wind speed in m/s" is a physical quantity independent of this engine's planet
// radius or cloud altitude, so boidish's tuning carries over directly): cloudFlowSpeed=0.25
// (both engines' default) means ~2.5 m/s, a gentle breeze. `h` is normalized
// height-in-cloud-band [0,1]; higher clouds shear ahead of/behind lower ones.
vec3 getCloudAdvectionSpeed(float h) {
	vec2  flowDir = vec2(cos(u_cloudFlowDirection), sin(u_cloudFlowDirection));
	float shear = 1.0 - (h * h) * u_cloudFlowHeightScale;
	return vec3(flowDir.x, 0.0, flowDir.y) * u_cloudFlowSpeed * shear * 10.0;
}

vec4 sampleCloudShadow(vec3 p) {
	return sampleCloudShadow(push.cloudShadowIndex, push.clipmapIndex, p, push.sunDir.xyz, uCameraPosition.xyz, 320000.0);
}

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

float heightFraction(vec3 worldPos, float distFromCam) {
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;

	return (adjustedY - 5000.0) / 15000.0;
}

// Define height gradients for different cloud types:
// vec4(bottom_fade_start, bottom_fade_end, top_fade_start, top_fade_end)
const vec4 stratusGrad       = vec4(0.0, 0.1, 0.2, 0.3);
const vec4 stratocumulusGrad = vec4(0.0, 0.1, 0.5, 0.7);
const vec4 cumulusGrad       = vec4(0.0, 0.1, 0.8, 1.0);

float GetDensityHeightGradientForPoint(float heightFraction, float cloudType) {
    float stratus = max(0.0, 1.0 - cloudType * 2.0);
    float stratocumulus = 1.0 - abs(cloudType - 0.5) * 2.0;
    float cumulus = max(0.0, (cloudType - 0.5) * 2.0);

    vec4 cloudGradient = stratusGrad * stratus +
                         stratocumulusGrad * stratocumulus +
                         cumulusGrad * cumulus;

    return smoothstep(cloudGradient.x, cloudGradient.y, heightFraction) -
           smoothstep(cloudGradient.z, cloudGradient.w, heightFraction);
}


// Single tileable detail volume (CloudDetailVolume, period CLOUD_VOLUME_PERIOD, all 3 axes) --
// replaces the former 3-cascade camera-distance blend. Vertical floor/ceiling shaping is the
// same curved-altitude gate the old cascades used; distFromCam only feeds that curvature.
float sampleCloudVolume(vec3 worldPos, float distFromCam) {
	float distSq = distFromCam * distFromCam;
	float dropOff = distSq / (2.0 * FAKE_PLANET_RADIUS);
	float adjustedY = worldPos.y + dropOff;

	float uv_y = (adjustedY - 5000.0) / 15000.0;

	// Fixes the vertical streaks (Screenshot 3) by preventing Y-axis wrapping
	if (uv_y < 0.0 || uv_y > 1.0) return 0.0;
	// worldPos += vec3(1,1.2,1)*500*cross_noise_fbm(vec3(0.005, 0.004, 0.005)*worldPos, 3, 0.3*uTime);
	vec3 uvw = fract(worldPos / CLOUD_VOLUME_PERIOD);
	uvw.y = fract(uv_y);

	// Distance-based LOD selection -- the other half of the alligator() noise fix (see
	// cloud_volume_gen.comp and CloudVolumeDownsampleNode): alligator's sharp Worley-cell edges
	// caused a ~10x GPU regression from neighboring raymarch threads disagreeing on threshold
	// branches when every sample read raw, unfiltered noise regardless of distance. Sampling a
	// pre-blurred mip at range (where per-voxel detail isn't visually distinguishable anyway)
	// tames that divergence. The distance curve below is a heuristic, not a derived screen-pixel
	// footprint (boidish's own calculatePixelWorldSizeToTextureLod does the latter, but that
	// needs a precise world-space-pixel-size term this raymarch doesn't carry) -- expect to
	// retune the 2000.0 reference distance visually. Clamped to 4 (not further) matching
	// boidish's own clamp(volumeMip, 0.0, 4.0) and CloudVolumeDesc()'s 5-mip chain.
	float volumeLod = clamp(log2(max(1.0, distFromCam / 2000.0)), 0.0, 4.0);
	return SAMPLE_3D_LINEAR_LOD(push.volumeIndex, uvw, volumeLod).r;
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

// Bifurcated step sizing, ported from boidish's atmosphere_lowres.comp
// (calculateBifurcatedStepSize): coarse strides through confirmed-empty space, a held-constant
// fine step while inside a cloud, and a step-back on entry so the boundary itself gets sampled
// at fine resolution instead of overshooting into the cloud on the first fine step. Replaces
// the old `stepSize *= 1.02` growth, which ran unconditionally every iteration -- including
// deep inside a thick cloud, where it should hold steady, not keep growing and under-sampling.
float calculateBifurcatedStepSize(
	inout bool inCloud,
	inout int  zeroCount,
	float      macroDensity,
	float      fineStep,
	float      cheapMult,
	int        escapeThreshold
) {
	float cheapStep = fineStep * cheapMult;

	if (!inCloud) {
		if (macroDensity > 0.01) {
			inCloud = true;
			zeroCount = 0;
			return -cheapStep;
		}
		return cheapStep;
	} else {
		if (macroDensity <= 0.0) {
			zeroCount++;
			if (zeroCount >= escapeThreshold) {
				inCloud = false;
				return cheapStep;
			}
		} else {
			zeroCount = 0;
		}
		return fineStep;
	}
}

// Reprojection extension point: once a temporal-reprojection pass exists, accumulate
// weightedDistSum/totalWeight (weight = cloudTransmittance * (1 - stepTransmittance), biased by
// exp(-accumulatedOpticalDepth * 15.0) so heavily-occluded-by-closer-cloud steps contribute less
// to the "representative" hit distance) the same way boidish's atmosphere_lowres.comp does, and
// return weightedDistSum/totalWeight as this march's depth/velocity estimate. Not added yet --
// nothing consumes it (no depth/velocity output target exists on CloudRenderNode today), and
// unused accumulation would just be optimized away rather than actually prepare anything.
// lightEnergy/transmittance are returned separately, not packed into a vec4's rgb/a, since
// transmittance is now vec3 (per-channel extinction, CloudLightingManager's cloudExtinctionColor)
// -- main()'s two-segment compositing already does its own manual blend (not a GPU fixed-
// function alpha blend), so there's no vec4-alpha constraint forcing transmittance to stay scalar.
struct CloudMarchResult {
	vec3 lightEnergy;
	vec3 transmittance;
};

// lightningPulseColor: rgb = brightest active strike's color, w = its normalized flicker
// envelope -- computed once per pixel in main() (see there for why), not per raymarch step.
CloudMarchResult marchClouds(
	vec3 worldRay, float t_start, float t_end, vec3 sunDir, vec3 sunRadiance, vec4 lightningPulseColor
) {
	float baseStepSize = 50.0;
	float t = t_start;

	vec2 fragCoord = gl_FragCoord.xy;
	float jitter = InterleavedGradientNoise(fragCoord, int(uFrameIndex));
	t += baseStepSize * jitter;

	vec3 cloudTransmittance = vec3(1.0);
	vec3 lightEnergy = vec3(0.0);
	float prevDensity = 0.0;

	bool inCloud = false;
	int zeroCount = 0;
	const float kCheapStepMult = 3.0; // boidish's own tuned value (atmosphere_lowres.comp:608)
	const int kEscapeThreshold = 3;

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

		float density = 0.0;
		float cloudHeightFrac = 0.0;
		if (weatherCoverage > 0.01) {
			cloudHeightFrac = heightFraction(p_cloud, distFromCam);
			weatherCoverage = pow(weatherCoverage, remapClamp(cloudHeightFrac, 0.7, 0.8, 1.0, mix(1.0, 0.5, weatherType)));

			// Wind advection applies to the fine detail volume only, not the weather-coverage
			// lookup above: the weather map already has its own emergent motion (the CA sim's
			// upwind-cell advection, helpers/wind.glsl), and drifting it a second time with an
			// independent speed/direction would fight that rather than add to it. cloudFlowSpeed
			// is specifically the fine-texture-drift knob, matching boidish's own separation of
			// "weather" advection from "3D noise" advection (computeCloudWeather vs
			// calculateCloudDensity in atmosphere_lowres.comp's source).
			vec3  advectedPos = p_cloud - getCloudAdvectionSpeed(clamp(cloudHeightFrac, 0.0, 1.0)) * uTime;
			float baseDensity = sampleCloudVolume(advectedPos, distFromCam);

// all(lessThan(cloudTransmittance, vec3(0.05)))
			float progress = smoothstep(0.015, 1.0, max(max(cloudTransmittance.r, cloudTransmittance.g), cloudTransmittance.b));
			mat3 grad;
			vec3 crNoise = cross_noise_fbm(advectedPos/75, 4, uTime*0.5, grad);
			vec3 cross_curl = get_curl(grad);
			// float baseDensity = weatherCoverage * sampleCloudCascades(p_cloud+100*progress*crNoise, distFromCam);
			// float density = 1.0 * baseDensity * (0.5 + 1.5 * weatherMoisture);

			// float dtNoise = dot_noise_fbm((p_cloud+25*progress*cross_curl)/750, 4, 0.05*uTime);
			// density = remapClamp(baseDensity, abs(dtNoise/4.0), 1, 0.0, 1);
			// density = 2*smoothstep(0.15, 0.85, density);


			// baseDensity = remapClamp(baseDensity, abs(dot_noise_fbm(advectedPos*0.000125, 3, 0.01*uTime)), 1, 0, 1);
			float heightGradient = GetDensityHeightGradientForPoint(cloudHeightFrac, weatherType);

			// Anvil spreading (boidish's calculateCloudDensity): strong convective
			// (cumulus-weighted) clouds flatten out near their top instead of tapering to a
			// point -- the classic cumulonimbus silhouette. weatherType doubles as boidish's
			// "type" axis here (0=stratus/1=cumulus, same semantic already sampled above), so
			// this needs no new baked per-type/height texture.
			float cumulusWeight = clamp(weatherType * 2.0 - 1.0, 0.0, 1.0); // >0 only in the cumulus half
			float anvilFactor = mix(1.0, 0.25, cumulusWeight * smoothstep(0.6, 0.9, cloudHeightFrac));
			float coverageThreshold = pow(1.0 - weatherCoverage, remapClamp(cloudHeightFrac, 0.6, 0.9, 1.0, anvilFactor));

			// Remap-based coverage erosion (boidish's calculateCloudDensity), not the old plain
			// multiply: treats coverageThreshold as a floor the shaped noise must clear before
			// any density survives, then ramps 0..1 above it. Gives crisp, billowy edges with
			// noise-driven boundary detail instead of the old multiply's uniformly-dimmed,
			// hazier look -- this is the single biggest shape difference from boidish's
			// reference, and the main thing this whole rework was chasing.
			float shapedNoise = remapClamp(baseDensity * heightGradient, coverageThreshold, 1, 0.0, 1.0);
			density = shapedNoise * (0.5 + 1.5 * weatherMoisture);

			// if(progress > 0) {
				// float dtNoise = progress*dot_noise_fbm((advectedPos+25*progress*cross_curl)/750, 4, 0.05*uTime);
				float dtNoise = dot_noise_fbm((advectedPos+25*cross_curl)/1500, 5, 0.05*uTime);
				density = remapClamp(baseDensity, abs(dtNoise/4.0), 1, 0.0, 1);
			// }
			density = 250*smoothstep(0.15, 0.85, density);

		}

		float nextStep =
			calculateBifurcatedStepSize(inCloud, zeroCount, density, baseStepSize, kCheapStepMult, kEscapeThreshold);

		if (nextStep < 0.0) {
			// Stepped back onto the cloud's entry boundary -- resample there next iteration
			// instead of doing lighting work from this (pre-entry, still-outside) position.
			t += nextStep;
			continue;
		}

		if (!inCloud) {
			prevDensity = density;
			t += nextStep;
			continue;
		}

		{
			float stepSize = nextStep; // held constant while inCloud, not grown
			if (density > 0.0) {
				// Type + height-profile extinction shaping (boidish's moistureExtinctionMult,
				// adapted to Brassica's available data -- no separate humidity channel, so
				// this layers on top of the existing moisture-coupled base rather than
				// replacing it): stratus (uniform, fog-like) clouds are optically denser per
				// unit shape than cumulus (chunky, sparser-but-denser-cored) at the same
				// density value; moisture pools near the cloud base, so extinction per unit
				// density falls off toward the top. Constants are boidish's own.
				float typeFactor = mix(1.2, 0.7, clamp(weatherType, 0.0, 1.0));
				float bottomMoistureProfile = clamp(1.0 - smoothstep(0.0, 0.85, cloudHeightFrac), 0.1, 1.0);
				bottomMoistureProfile *= bottomMoistureProfile;
				float extinction = max(
					0.0, density * (0.005 + 0.02 * weatherMoisture) * typeFactor * (0.6 + 0.8 * bottomMoistureProfile)
				);
				// Per-channel extinction (CloudLightingManager's cloudExtinctionScale/Color) --
				// boidish's rgbExtinction, multiplied through the whole optical-depth chain
				// below (stepDensity/stepOpticalDepth/transmittance/the self-shadow term/the
				// OCTAVES multi-scatter loop), not just a final tint.
				vec3 rgbExtinction = extinction * u_cloudExtinctionColor * u_cloudExtinctionScale;
				float averageDensity = (prevDensity + density) * 0.5;
				vec3  stepDensity = averageDensity * rgbExtinction;
				vec3  stepOpticalDepth = stepDensity * stepSize;
				vec3  stepTransmittance = exp(-stepOpticalDepth);

				vec4 cloudShadowData = sampleCloudShadow(p_cloud);
				vec3 opticalDepthToLight = cloudShadowData.b * u_cloudShadowOpticalDepthMultiplier * rgbExtinction;

				float r_km = r_world / 1000.0;
                float mu = dot(surfaceDir, sunDir);
                vec3 atmosphereTransmittance = getTransmittance(r_km, mu);

				float cosTheta = dot(worldRay, sunDir);
				const int OCTAVES = 3;
				float currentExtinction = 1.0;
				float currentKfwd = u_cloudPhaseG1;
				float currentKbck = u_cloudPhaseG2;
				float currentEnergy = 1.0;
				float currentScatChaos = (1.0 - u_cloudPhaseIsotropic);

				vec3 msScattering = vec3(0.0);

				for(int oct = 0; oct < OCTAVES; oct++) {
					float phase = dualLobeSchlick(cosTheta, currentKfwd, currentKbck, u_cloudPhaseAlpha);
					phase = mix((1.0 / (4.0 * PI)), phase, currentScatChaos);

					vec3 octaveOpticalDepth = opticalDepthToLight * currentExtinction;
					vec3 octaveStepDensity = stepDensity * currentExtinction;

					vec3 shadowTerm = mix(
						beerPowder(
							octaveOpticalDepth, octaveStepDensity, u_directBeerPowderScale, u_cloudPowderMultiplier, u_cloudPowderLocalScale
						),
						exp(-octaveOpticalDepth),
						u_cloudBeerPowderMix
					);

					msScattering += shadowTerm * phase * currentEnergy;

					currentExtinction *= 0.5;
					currentKfwd *= 0.5;
					currentKbck *= 0.5;
					currentEnergy *= 0.5;
					currentScatChaos *= 0.5;
				}

				// --- Ground Upwelling ---
				vec3 averageTerrainAlbedo = vec3(0.18, 0.16, 0.12);
				vec3 upwellingBase = sunRadiance * atmosphereTransmittance * max(0.0, sunDir.y) * averageTerrainAlbedo * (1.0 / 3.14159265);
				vec3 upwellingOpticalDepth = stepDensity * clamp(cloudHeightFrac, 0.0, 1.0) * 15000.0;
				vec3 groundUpwelling = upwellingBase * exp(-upwellingOpticalDepth * 0.1);

				// --- Backscatter & Top Diffuse Illumination ---
				// When viewed from above (cosTheta < 0, looking down), direct sunlight entering the top of the cloud
				// undergoes multiple scattering near the top surface and is diffusely backscattered out towards the camera.
				float topLighting = smoothstep(0.1, 0.9, cloudHeightFrac);
				float backscatterWeight = max(0.0, -cosTheta) * topLighting;
				float isotropicPhase = 1.0 / (4.0 * 3.14159265);
				vec3 diffuseTopBackscatter = sunRadiance * atmosphereTransmittance * max(0.0, sunDir.y) * isotropicPhase * (0.8 + 1.2 * backscatterWeight) * exp(-opticalDepthToLight);

				vec3 powder = vec3(1.0) - vec3(1.0) / (vec3(1.0) + (25.0 * stepDensity * u_ambientBeerPowderScale));

				// Directional ambient (helpers/ambient.glsl) standing in for a real SH probe
				// grid -- LightingUBO already reserves uSHCoeffs[81] for one, nothing bakes it
				// yet. groundUpwelling (above) is the sun-reactive part of the ground term and
				// is kept as-is; the approximation just supplies the ground/sky/horizon floor
				// a flat, no-sun ambient would otherwise be missing.
				vec3 skyAmbient = getApproximateSkyAmbient(sunDir, sunRadiance);
				vec3 groundAmbient = getApproximateGroundAmbient(sunRadiance) + groundUpwelling;
				vec3 horizonAmbient = getApproximateHorizonAmbient(sunDir, sunRadiance, skyAmbient, groundAmbient);
				vec3 ambientTop = mix(skyAmbient, horizonAmbient, 0.4);

				vec3 upwardOD = stepDensity * clamp(cloudHeightFrac, 0.0, 1.0) * 15000.0;
				vec3 downwardOD = stepDensity * clamp(1.0 - cloudHeightFrac, 0.0, 1.0) * 15000.0;
				vec3 skyAttenuation = vec3(1.0) / (vec3(1.0) + downwardOD);
				vec3 groundAttenuation = vec3(1.0) / (vec3(1.0) + upwardOD);

				vec3 ambientGlow = ((ambientTop * skyAttenuation) + (groundAmbient * groundAttenuation)) * powder;

				// Lightning emissive glow: previously read uLightningColor (shaders/lighting.glsl's
				// LightingUBO), fed by the CPU LightningManager -- which has zero production
				// callers, so that was always zero in practice. Now reads lightningPulseColor
				// (the GPU-driven strike pool's brightest active slot, see main()) instead.
				// weatherSample.a (lightning propensity, already sampled above) stands in for
				// boidish's per-point "ridge" noise, tying the propagating flash's structure to
				// where lightning is actually likely instead of an arbitrary noise channel.
				vec3  emissiveGlow = vec3(0.0);
				float lightningPulse = lightningPulseColor.w;
				if (lightningPulse > 0.001) {
					float ridge = weatherSample.a;
					float wavePos = clamp(1.0 - lightningPulse, 0.0, 1.0);
					float midHeightEnvelope = exp(-pow((cloudHeightFrac - 0.5) * 4.0, 2.0));
					float temporalPhase = fract(uTime * 0.7 + ridge * 3.0);
					float stagger = sin(temporalPhase * 6.28318) * 0.5 + 0.5;
					float ridgeDist = abs(ridge - wavePos * 0.85);
					float waveFront = exp(-ridgeDist * ridgeDist * 16.0);
					float coreGlow = exp(-ridge * 4.0 / max(0.1, lightningPulse));
					float propagationGlow = (coreGlow * 0.7 + waveFront * 1.3) * lightningPulse * midHeightEnvelope * stagger;
					emissiveGlow = lightningPulseColor.rgb * propagationGlow;
				}

				// cloudMoonLightScale (CloudLightingManager) is exposed but not yet wired here --
				// this march only has a single incoming light (sunDir/sunRadiance, passed by
				// main()); moon support would need the second light threaded through the same
				// way, which doesn't exist yet.
				vec3 directLight = sunRadiance * atmosphereTransmittance * msScattering * u_cloudSunLightScale;
				// cloudAlbedo (CloudLightingManager) scales reflected light only -- emissiveGlow
				// is self-luminous (lightning), not reflective, so it's added back in afterward
				// rather than scaled by albedo too (boidish's own structure: S = (stepScattering
				// + ambient) * albedo + emissiveGlow, not (... + emissiveGlow) * albedo).
				vec3 reflectedLight = (directLight + ambientGlow + diffuseTopBackscatter * powder) * u_cloudAlbedo;
				vec3 stepScattering = reflectedLight + emissiveGlow;

				// FIX: Removed the `/ max(stepDensity, 0.0001)` that was blowing up the light
				vec3 inscatter = stepScattering * (1.0 - stepTransmittance);

				lightEnergy += inscatter * cloudTransmittance;
				cloudTransmittance *= stepTransmittance;

				// all(), not any(): a channel with lower extinction can still be meaningfully
				// transmissive even once another channel has already dropped below threshold,
				// and cutting the march short there would visibly discolor the result.
				if (all(lessThan(cloudTransmittance, vec3(0.05)))) break;
			}
			prevDensity = density;
		}

		t += nextStep;
	}

	// cloudColor (CloudLightingManager): final flat tint on the accumulated light, applied once
	// here rather than per-segment at the call site (boidish applies it the same way, as the
	// very last step before returning).
	return CloudMarchResult(lightEnergy * u_cloudColor, cloudTransmittance);
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

		// Brightest active GPU-driven strike (shaders/lighting.glsl's reserved uLights[] range,
		// written by lightning_age.comp) -- computed once per pixel, not per raymarch step
		// inside marchClouds(), since it's the same value for the whole screen. Un-scales
		// intensity back to the normalized-ish [0,1] flicker envelope marchClouds()'s glow
		// formula expects (intensity itself is scaled by u_lightningFlashIntensity * 50.0).
		vec4  lightningPulseColor = vec4(0.0);
		float lightningScaleInv = 1.0 / max(0.0001, u_lightningFlashIntensity * 50.0);
		for (uint li = 0u; li < LIGHTNING_MAX_STRIKES; li++) {
			float strikePulse = uLights[LIGHTNING_LIGHT_SLOT_BASE + li].intensity * lightningScaleInv;
			if (strikePulse > lightningPulseColor.w) {
				lightningPulseColor = vec4(uLights[LIGHTNING_LIGHT_SLOT_BASE + li].color, strikePulse);
			}
		}

		// Manual (not GPU fixed-function alpha) blend -- transmittance is vec3 (per-channel
		// extinction, CloudLightingManager's cloudExtinctionColor), so it's tracked separately
		// rather than packed into a vec4's alpha the way a single-channel transmittance was.
		vec3 resultRgb = vec3(0.0);
		vec3 resultTransmittance = vec3(1.0);

		// Segment 1 (Near side of the cloud shell)
		if (t_s1 < t_e1 && t_s1 < surfaceDistKM) {
			CloudMarchResult near =
				marchClouds(worldRay, t_s1, min(t_e1, surfaceDistKM), sunDir, sunRadiance, lightningPulseColor);
			resultRgb += near.lightEnergy;
			resultTransmittance *= near.transmittance;
		}

		// Segment 2 (Far side of the cloud shell, if ray pierces entirely through space)
		if (t_s2 < t_e2 && t_s2 < surfaceDistKM) {
			CloudMarchResult far =
				marchClouds(worldRay, t_s2, min(t_e2, surfaceDistKM), sunDir, sunRadiance, lightningPulseColor);
			resultRgb += far.lightEnergy * resultTransmittance;
			resultTransmittance *= far.transmittance;
		}

		currentRadiance = currentRadiance * resultTransmittance + resultRgb;
	}
	outColor = vec4(currentRadiance, 1.0);
}
