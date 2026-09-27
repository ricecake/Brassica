#ifndef HELPERS_CLOUD_UTILS_GLSL
#define HELPERS_CLOUD_UTILS_GLSL

#include "../atmosphere/common.glsl"
#include "../bindless.glsl"
#include "../textures/cloud.glsl"

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

float remap(float value, float valueMin, float valueMax) {
	return (value - valueMin) / max(1e-5, (valueMax - valueMin));
}

float remapClamp(float value, float inMin, float inMax, float outMin, float outMax) {
	float t = clamp((value - inMin) / max(1e-5, (inMax - inMin)), 0.0, 1.0);
	return mix(outMin, outMax, t);
}

float schlickGain(float x, float g) {
	g = clamp(g, 0.001, 0.999);
	float absDiff = abs(2.0 * x - 1.0);
	float denominator = g + absDiff * (1.0 - 2.0 * g);
	return 0.5 + ((x - 0.5) * (1.0 - g)) / denominator;
}

float cloudPhase(float cosTheta, float phaseG1, float phaseG2, float phaseAlpha, float phaseIsotropic) {
	float hg = mix(henyeyGreenstein(phaseG1, cosTheta), henyeyGreenstein(phaseG2, cosTheta), phaseAlpha);
	return mix(hg, (1.0 / (4.0 * PI)), phaseIsotropic);
}

float beerPowder(float d, float local_d, float powderScale, float powderMultiplier) {
	return max(
		exp(-d),
		exp(-d * powderScale) * powderMultiplier * (1.0 - exp(-local_d * 5.0))
	);
}

vec3 beerPowder(vec3 d, vec3 local_d, float powderScale, float powderMultiplier) {
	return max(
		exp(-d),
		exp(-d * powderScale) * powderMultiplier * (vec3(1.0) - exp(-local_d * 5.0))
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

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, float cloudAltitude, float cloudThickness, out float t_start1, out float t_end1, out float t_start2, out float t_end2) {
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

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, float cloudAltitude, float cloudThickness, out float t_start, out float t_end) {
	float t_s1, t_e1, t_s2, t_e2;
	if (intersectCloudShell(ro, rd, worldScaleVal, cloudAltitude, cloudThickness, t_s1, t_e1, t_s2, t_e2)) {
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

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, out float t_start1, out float t_end1, out float t_start2, out float t_end2) {
	return intersectCloudShell(ro, rd, worldScaleVal, 2000.0, 1500.0, t_start1, t_end1, t_start2, t_end2);
}

bool intersectCloudShell(vec3 ro, vec3 rd, float worldScaleVal, out float t_start, out float t_end) {
	return intersectCloudShell(ro, rd, worldScaleVal, 2000.0, 1500.0, t_start, t_end);
}

float getCurvedAltitude(vec3 p, float worldScaleVal) {
	float R_earth = kEarthRadius * 1000.0 * worldScaleVal;
	vec3 earthCenter = vec3(uCameraPosition.x, -R_earth, uCameraPosition.z);
	return length(p - earthCenter) - R_earth;
}

float getCloudRelativeHeight(vec3 p, CloudWeather weather, float worldScaleVal, out float localFloor, out float actualThickness) {
	float altitude = getCurvedAltitude(p, worldScaleVal);
	float altitudeShift = weather.heightMap * weather.height;

	actualThickness = max(weather.thickness * weather.height, 25.0 * worldScaleVal);
	localFloor = weather.baseFloor + altitudeShift;
	return clamp((altitude - localFloor) / actualThickness, 0.0, 1.0);
}

float getCloudRelativeHeight(vec3 p, CloudWeather weather, float worldScaleVal) {
	float localFloor, actualThickness;
	return getCloudRelativeHeight(p, weather, worldScaleVal, localFloor, actualThickness);
}

float getCloudRelativeHeight(vec3 p, CloudWeather weather) {
	return getCloudRelativeHeight(p, weather, 1.0);
}

vec3 getCloudWindSpeed(float timeVal, float flowDirRad, float flowSpeed, float worldScaleVal) {
	vec2 flowDir = vec2(cos(flowDirRad), sin(flowDirRad));
	return vec3(flowDir.x, 0.0, flowDir.y) * flowSpeed * worldScaleVal * 10.0;
}

vec3 getCloud3DNoiseAdvectionSpeed(float h, float timeVal, float flowDirRad, float flowSpeed, float worldScaleVal) {
	vec2 flowDir = vec2(cos(flowDirRad), sin(flowDirRad));
	float shear = 1.0 - (h * h) * 0.015 * 1.0;

	vec3 noiseSpeed = -vec3(flowDir.x, 0.05, flowDir.y) * (flowSpeed * 0.75) * worldScaleVal * 10.0;
	noiseSpeed.xz += flowDir * shear * worldScaleVal * 10.0;
	return noiseSpeed;
}

vec3 getCloudAdvectionSpeed(float h, float timeVal, float flowDirRad, float flowSpeed, float worldScaleVal) {
	return getCloudWindSpeed(timeVal, flowDirRad, flowSpeed, worldScaleVal) + getCloud3DNoiseAdvectionSpeed(h, timeVal, flowDirRad, flowSpeed, worldScaleVal);
}

vec3 getCloudAdvectionSpeed(float h, float timeVal) {
	return getCloudAdvectionSpeed(h, timeVal, 3.14159265, 0.25, 1.0);
}

vec3 getCloudWindOffset(float timeVal, float worldScaleVal) {
	return timeVal * getCloudWindSpeed(timeVal, 3.14159265, 0.25, worldScaleVal);
}

vec3 getCloudWindOffset(float timeVal) {
	return getCloudWindOffset(timeVal, 1.0);
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

CloudWeather computeCloudWeather(vec3 p, CloudProperties props, float lod, uint weatherTexIdx, float timeVal) {
	vec3 advect = getCloudWindOffset(timeVal, props.worldScale);
	vec3 p_advected = p - advect;

	vec2 uv = p_advected.xz / (100000.0 * props.worldScale);
	vec4 bakedWeather = textureLod(sampler2D(uTextures2D[nonuniformEXT(weatherTexIdx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv, clamp(lod, 0.0, 11.0));

	return loadCloudWeather(p, props, bakedWeather);
}

CloudWeather computeCloudWeather(vec3 p, CloudProperties props, uint weatherTexIdx, float timeVal) {
	return computeCloudWeather(p, props, 0.0, weatherTexIdx, timeVal);
}

float sampleDeepOpacityMap(uint shadowMapIdx, vec2 shadowUV, float h, float lod) {
	float layerIdx = 8.0 * (1.0 - h) - 1.0;
	if (layerIdx < 0.0) {
		float t = layerIdx + 1.0;
		float depth0 = SAMPLE_ARRAY_LINEAR_LOD(shadowMapIdx, vec3(shadowUV, 0.0), lod).r;
		return mix(0.0, depth0, clamp(t, 0.0, 1.0));
	} else {
		float floorIdx = floor(layerIdx);
		float ceilIdx = ceil(layerIdx);
		float t = fract(layerIdx);
		float depthFloor = SAMPLE_ARRAY_LINEAR_LOD(shadowMapIdx, vec3(shadowUV, clamp(floorIdx, 0.0, 7.0)), lod).r;
		float depthCeil = SAMPLE_ARRAY_LINEAR_LOD(shadowMapIdx, vec3(shadowUV, clamp(ceilIdx, 0.0, 7.0)), lod).r;
		return mix(depthFloor, depthCeil, t);
	}
}

float calculateCloudShadowFactor(vec3 frag_pos, vec3 L, float intensity, uint shadowMapIdx, uint weatherTexIdx, float timeVal) {
	if (intensity <= 0.0) return 1.0;

	CloudProperties props;
	props.altitude = 2000.0;
	props.thickness = 1500.0;
	props.densityBase = 0.100;
	props.coverage = 0.35;
	props.worldScale = 1.0;

	CloudWeather weather = computeCloudWeather(frag_pos, props, weatherTexIdx, timeVal);
	float h = getCloudRelativeHeight(frag_pos, weather, props.worldScale);

	vec2 shadowUV = frag_pos.xz * 0.0001; // project
	float accumulatedDensity = sampleDeepOpacityMap(shadowMapIdx, shadowUV, h, 0.0) * 0.001 * 2.0 / max(0.001, props.worldScale);
	float shadowTerm = exp(-accumulatedDensity);

	return mix(1.0, shadowTerm, intensity);
}

#endif // HELPERS_CLOUD_UTILS_GLSL
