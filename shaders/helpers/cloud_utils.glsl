#ifndef HELPERS_CLOUD_UTILS_GLSL
#define HELPERS_CLOUD_UTILS_GLSL

#include "../atmosphere/common.glsl"
#include "../textures/cloud.glsl"

#ifndef CLOUD_SHADOW_MAP_BINDING
#define CLOUD_SHADOW_MAP_BINDING 10
#endif

layout(binding = CLOUD_SHADOW_MAP_BINDING) uniform sampler2DArray u_cloudShadowTexture;

layout(std140, set = 0, binding = 5) uniform CloudParamsUBO {
	mat4  u_cloudShadowMatrix;
	bool  u_useCloudShadowMap;

	float cloudAltitude;
	float cloudThickness;
	float cloudDensity;
	float cloudCoverage;
	float worldScale;

	float cloudPhaseG1;
	float cloudPhaseG2;
	float cloudPhaseAlpha;
	float cloudPhaseIsotropic;
	float cloudPowderScale;
	float cloudPowderMultiplier;
	float cloudPowderLocalScale;
	float cloudBeerPowderMix;

	float cloudShadowOpticalDepthMultiplier;
	float cloudShadowStepMultiplier;
	float cloudShadowIntensity;
	float cloudSunLightScale;
	float cloudMoonLightScale;

	float cloudFlowSpeed;
	float cloudFlowDirection;
	float cloudFlowHeightScale;
	float cloudCurlStrength;
	float cloudCurlFrequency;
};

#define u_time uTime

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
	float heightMap;
	float thickness;
	float ridge;
	float ecentricity;
	float curve;
	float centerDist;
	float baseFloor;
	float baseCeiling;
	float height;
	float moisture;
	float humidity;
};


float cloudPhase(float cosTheta) {
	float hg = mix(henyeyGreenstein(cloudPhaseG1, cosTheta), henyeyGreenstein(cloudPhaseG2, cosTheta), cloudPhaseAlpha);
	return mix(hg, (1.0 / (4.0 * PI)), cloudPhaseIsotropic);
}

float beerPowder(float d, float local_d) {
	return max(
		exp(-d),
		exp(-d * cloudPowderScale) * cloudPowderMultiplier * (1.0 - exp(-local_d * cloudPowderLocalScale))
	);
}

vec3 beerPowder(vec3 d, vec3 local_d) {
	return max(
		exp(-d),
		exp(-d * cloudPowderScale) * cloudPowderMultiplier * (vec3(1.0) - exp(-local_d * cloudPowderLocalScale))
	);
}

bool intersectSphereLocal(vec3 ro, vec3 rd, float radius, out float t0, out float t1) {
	float b = dot(ro, rd);
	float rLen = length(ro);
	float c = (rLen - radius) * (rLen + radius);
	float det = b * b - c;
	if (det < 0.0)
		return false;
	det = sqrt(det);
	t0 = -b - det;
	t1 = -b + det;
	return true;
}

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, out float t_start1, out float t_end1, out float t_start2, out float t_end2) {
	float R_earth = kEarthRadius * 1000.0 * worldScaleVal;
	float R_floor = R_earth + (cloudAltitude - 500.0) * worldScaleVal;
	float R_ceiling = R_earth + (cloudAltitude + 2.0 * cloudThickness + 500.0) * worldScaleVal;

	vec3 earthCenter = vec3(uCameraPosition.x, -R_earth, uCameraPosition.z);
	vec3 relRo = ro - earthCenter;

	t_start1 = 0.0;
	t_end1 = -1.0;
	t_start2 = 0.0;
	t_end2 = -1.0;

	float t_out0, t_out1;
	if (!intersectSphereLocal(relRo, rd, R_ceiling, t_out0, t_out1) || t_out1 <= 0.0) {
		return false;
	}

	float t_e0, t_e1;
	bool hits_earth = intersectSphereLocal(relRo, rd, R_earth, t_e0, t_e1) && (t_e0 > 0.0);
	float t_max = hits_earth ? t_e0 : t_out1;

	float t_c_entry = max(0.0, t_out0);
	float t_c_exit  = min(t_out1, t_max);

	if (t_c_entry >= t_c_exit) {
		return false;
	}

	float t_in0, t_in1;
	if (intersectSphereLocal(relRo, rd, R_floor, t_in0, t_in1)) {
		if (t_in0 < 0.0) {
			float ts = max(t_c_entry, t_in1);
			float te = t_c_exit;
			if (ts < te) {
				t_start1 = ts;
				t_end1 = te;
				return true;
			}
			return false;
		} else {
			float ts1 = t_c_entry;
			float te1 = min(t_c_exit, t_in0);
			if (ts1 < te1) {
				t_start1 = ts1;
				t_end1 = te1;
			}

			float ts2 = max(t_c_entry, t_in1);
			float te2 = t_c_exit;
			if (ts2 < te2) {
				if (t_start1 >= t_end1) {
					t_start1 = ts2;
					t_end1 = te2;
				} else {
					t_start2 = ts2;
					t_end2 = te2;
				}
			}

			return (t_start1 < t_end1) || (t_start2 < t_end2);
		}
	} else {
		t_start1 = t_c_entry;
		t_end1 = t_c_exit;
		return t_start1 < t_end1;
	}
}

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, out float t_start, out float t_end) {
	float t_s1, t_e1, t_s2, t_e2;
	if (intersectCloudShell(ro, rd, worldScaleVal, t_s1, t_e1, t_s2, t_e2)) {
		if (t_s1 < t_e1) {
			t_start = t_s1;
			t_end = t_e1;
			return true;
		} else if (t_s2 < t_e2) {
			t_start = t_s2;
			t_end = t_e2;
			return true;
		}
	}
	t_start = 1e10;
	t_end = -1e10;
	return false;
}

float getCurvedAltitude(vec3 p) {
	float R_earth = kEarthRadius * 1000.0 * worldScale;
	vec3 earthCenter = vec3(uCameraPosition.x, -R_earth, uCameraPosition.z);
	return length(p - earthCenter) - R_earth;
}

float getCloudRelativeHeight(vec3 p, CloudWeather weather, out float localFloor, out float actualThickness) {
	float altitude = getCurvedAltitude(p);
	float altitudeShift = weather.heightMap * weather.height;

	actualThickness = max(weather.thickness * weather.height, 25.0 * worldScale);
	localFloor = weather.baseFloor + altitudeShift;
	return clamp((altitude - localFloor) / actualThickness, 0.0, 1.0);
}

float getCloudRelativeHeight(vec3 p, CloudWeather weather) {
	float localFloor, actualThickness;
	return getCloudRelativeHeight(p, weather, localFloor, actualThickness);
}

vec3 getCloudWindSpeed(float timeVal) {
	float angle = cloudFlowDirection;
	vec2  flowDir = vec2(cos(angle), sin(angle));
	return vec3(flowDir.x, 0.0, flowDir.y) * cloudFlowSpeed * worldScale * 10.0;
}

vec3 getCloud3DNoiseAdvectionSpeed(float h, float timeVal) {
	float angle = cloudFlowDirection;
	vec2  flowDir = vec2(cos(angle), sin(angle));
	float shear = 1.0 - (h * h) * cloudFlowHeightScale * 1.0;

	vec3 noiseSpeed = -vec3(flowDir.x, 0.05, flowDir.y) * (cloudFlowSpeed * 0.75) * worldScale * 10.0;
	noiseSpeed.xz += flowDir * shear * worldScale * 10.0;
	return noiseSpeed;
}

vec3 getCloudAdvectionSpeed(float h, float timeVal) {
	return getCloudWindSpeed(timeVal) + getCloud3DNoiseAdvectionSpeed(h, timeVal);
}

vec3 getCloudWindOffset(float timeVal) {
	return timeVal * getCloudWindSpeed(timeVal);
}

float applyDynamicCoverage(float bakedCoverage, float uniformCoverage) {
	float coverageFloor = 1.0 - (uniformCoverage * 2.0);
	float remapped = clamp((bakedCoverage - coverageFloor) / max(1e-5, (1.0 - min(0.0, coverageFloor))), 0.0, 1.0);
	return schlickGain(remapped, 0.25);
}

CloudWeather loadCloudWeather(vec3 p, CloudProperties props, vec4 tex, vec4 frontSample) {
	CloudWeather weather;
	weather.p = p;

	float baseCoverage = tex.r;
	float frontCoverageBoost = frontSample.g;
	weather.coverage = applyDynamicCoverage(baseCoverage * frontCoverageBoost, props.coverage);

	float bakedType = mix(0.05, 0.75, tex.g);
	weather.heightMap = mix(bakedType, frontSample.r, 0.5);

	float frontThicknessMod = frontSample.b;
	weather.thickness = mix(0.15, 1.0, tex.g) * frontThicknessMod;
	weather.density = tex.a * props.densityBase * frontCoverageBoost;
	weather.ridge = tex.b;
	weather.moisture = tex.a;
	weather.humidity = frontSample.b;
	weather.ecentricity = frontSample.a;

	weather.baseFloor = props.altitude * props.worldScale;
	weather.height = max(props.thickness * frontThicknessMod, 0.001) * props.worldScale;
	weather.baseCeiling = weather.baseFloor + 2.0 * weather.height;

	return weather;
}

CloudWeather loadCloudWeather(vec3 p, CloudProperties props, vec4 tex) {
	return loadCloudWeather(p, props, tex, vec4(0.5, 1.0, 1.0, 1.0));
}

CloudWeather computeCloudWeather(vec3 p, CloudProperties props, float lod) {
	vec3 advect = getCloudWindOffset(uTime);
	vec3 p_advected = p - advect;

	vec2 uv = p_advected.xz / (100000.0 * props.worldScale);
	vec4 bakedWeather = textureLod(u_cloudWeatherTexture, uv, clamp(lod, 0.0, 11.0));

	return loadCloudWeather(p, props, bakedWeather);
}

CloudWeather computeCloudWeather(vec3 p, CloudProperties props) {
	return computeCloudWeather(p, props, 0.0);
}

float getDistanceToCloudEdge(float coverage, float h, float type, float thicknessVal) {
	float horizontalScale = max(thicknessVal * 2.0, 2000.0);
	float distHorizontal = coverage * horizontalScale;
	float distVertical = min(h, 1.0 - h) * thicknessVal;
	return max(0.0, min(distHorizontal, distVertical));
}

float sampleDeepOpacityMap(vec2 shadowUV, float h, float lod) {
	float layerIdx = 8.0 * (1.0 - h) - 1.0;
	if (layerIdx < 0.0) {
		float t = layerIdx + 1.0;
		float depth0 = textureLod(u_cloudShadowTexture, vec3(shadowUV, 0.0), lod).r;
		return mix(0.0, depth0, clamp(t, 0.0, 1.0));
	} else {
		float floorIdx = floor(layerIdx);
		float ceilIdx = ceil(layerIdx);
		float t = fract(layerIdx);
		float depthFloor = textureLod(u_cloudShadowTexture, vec3(shadowUV, clamp(floorIdx, 0.0, 7.0)), lod).r;
		float depthCeil = textureLod(u_cloudShadowTexture, vec3(shadowUV, clamp(ceilIdx, 0.0, 7.0)), lod).r;
		return mix(depthFloor, depthCeil, t);
	}
}

float calculateCloudShadowFactor(vec3 frag_pos, vec3 L, float intensity) {
	if (intensity <= 0.0) return 1.0;
	if (!u_useCloudShadowMap) return 1.0;

	vec4 lightSpacePos = u_cloudShadowMatrix * vec4(frag_pos, 1.0);
	vec2 shadowUV = lightSpacePos.xy * 0.5 + 0.5;

	CloudProperties props;
	props.altitude = cloudAltitude;
	props.thickness = cloudThickness;
	props.densityBase = cloudDensity;
	props.coverage = cloudCoverage;
	props.worldScale = worldScale;

	CloudWeather weather = computeCloudWeather(frag_pos, props);
	float h = getCloudRelativeHeight(frag_pos, weather);

	float accumulatedDensity = sampleDeepOpacityMap(shadowUV, h, 0.0) * 0.001 * cloudShadowOpticalDepthMultiplier / max(0.001, worldScale);
	float shadowTerm = exp(-accumulatedDensity);

	return mix(1.0, shadowTerm, intensity * cloudShadowIntensity);
}

float calculateCloudAmbientOcclusion(vec3 frag_pos) {
	if (!u_useCloudShadowMap) return 1.0;

	CloudProperties props;
	props.altitude = cloudAltitude;
	props.thickness = cloudThickness;
	props.densityBase = cloudDensity;
	props.coverage = cloudCoverage;
	props.worldScale = worldScale;

	CloudWeather weather = computeCloudWeather(frag_pos, props);

	float localFloor, actualThickness;
	float h = getCloudRelativeHeight(frag_pos, weather, localFloor, actualThickness);

	float alt_above = clamp(getCurvedAltitude(frag_pos), localFloor, localFloor + actualThickness);
	float R_earth = kEarthRadius * 1000.0 * worldScale;
	float distXZ_sq = dot(frag_pos.xz - uCameraPosition.xz, frag_pos.xz - uCameraPosition.xz);
	float rad_above = alt_above + R_earth;
	float y_above = sqrt(max(0.0, rad_above * rad_above - distXZ_sq)) - R_earth;
	vec3 P_above = vec3(frag_pos.x, y_above, frag_pos.z);

	vec4 lightSpacePos_above = u_cloudShadowMatrix * vec4(P_above, 1.0);
	vec2 shadowUV_above = lightSpacePos_above.xy * 0.5 + 0.5;

	float accumulatedDensity = sampleDeepOpacityMap(shadowUV_above, h, 1.0) * 0.001 * cloudShadowOpticalDepthMultiplier / max(0.001, worldScale);
	float cloudAO = exp(-accumulatedDensity);

	return mix(1.0, cloudAO, cloudShadowIntensity);
}

#endif // HELPERS_CLOUD_UTILS_GLSL
