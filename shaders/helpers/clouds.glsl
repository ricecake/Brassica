#ifndef HELPERS_CLOUDS_GLSL
#define HELPERS_CLOUDS_GLSL

#include "../lighting.glsl"
#include "cloud_utils.glsl"

#ifndef CLOUD_3D_VOLUME_BINDING
#define CLOUD_3D_VOLUME_BINDING 11
#endif

layout(binding = CLOUD_3D_VOLUME_BINDING) uniform sampler3D u_cloud3DTexture;

struct CloudSpotDetails {
	float density;
	vec3 relativeExtinction;
	vec3 advectionSpeed;
};

struct CloudDensityResult {
	vec3 density;
	vec3 advectionSpeed;
	float ao;
	vec3 albedo;
	vec3 emissivity;
	vec3 relativeExtinction;
};

CloudSpotDetails calculateCloudDensity(
	vec3            p,
	CloudWeather    weather,
	CloudProperties props,
	float           timeVal,
	float           simplified,
	bool            doCheap,
	vec4            volNoises
) {
	float localFloor, actualThickness;
	float h = getCloudRelativeHeight(p, weather, localFloor, actualThickness);

	vec3 advectSpeed = getCloudAdvectionSpeed(h, timeVal);
	float baseNoise = volNoises.r;

	float anvilFactor = mix(1.0, 0.25, 0.5);
	float cloud_coverage = pow(clamp(1.0 - weather.coverage, 0.0, 1.0), remapClamp(h, 0.6, 0.9, 1.0, anvilFactor));
	baseNoise = remapClamp(baseNoise, cloud_coverage, 1.0, 0.0, 1.0);

	if (!doCheap) {
		baseNoise = remapClamp(baseNoise, 0.0, 1.0, 0.0, 0.85);
	}

	float bottomMoistureProfile = clamp(1.0 - smoothstep(0.0, 0.85, h), 0.1, 1.0);
	bottomMoistureProfile = bottomMoistureProfile * bottomMoistureProfile;

	float typeFactor = mix(1.2, 0.7, clamp(weather.heightMap, 0.0, 1.0));
	float moistureExtinctionMult = (0.5 + 1.5 * weather.moisture) * (0.8 + 0.6 * weather.humidity) * typeFactor * (0.6 + 0.8 * bottomMoistureProfile);

	return CloudSpotDetails(
		clamp(baseNoise, 0.0, 1.0),
		vec3(moistureExtinctionMult),
		advectSpeed
	);
}

CloudDensityResult calculateCloudDensity(
	vec3            p,
	CloudWeather    weather,
	CloudProperties props,
	float           timeVal,
	float           lod,
	bool            doCheap
) {
	float localFloor, actualThickness;
	float h = getCloudRelativeHeight(p, weather, localFloor, actualThickness);
	vec3 advectSpeed = getCloudAdvectionSpeed(h, timeVal);
	CloudDensityResult pointDetails = CloudDensityResult(vec3(0.0), advectSpeed, 1.0, vec3(1.0), vec3(0.0), vec3(1.0));

	if (p.y < localFloor || p.y > (localFloor + actualThickness)) {
		return pointDetails;
	}

	vec3 noiseAdvectSpeed = getCloudAdvectionSpeed(h, timeVal);
	vec3 advect_3d = timeVal * noiseAdvectSpeed;
	vec3 p_advected_3d = p - advect_3d;

	float volumeScale = 7000.0 * props.worldScale;
	vec3 uvw = p_advected_3d / volumeScale;

	float texelWorldSize = volumeScale / max(1.0, float(textureSize(u_cloud3DTexture, 0).x));
	float volumeMip = lod;

	vec4 volSample = textureLod(u_cloud3DTexture, uvw, clamp(volumeMip, 0.0, 4.0));

	CloudSpotDetails res = calculateCloudDensity(p, weather, props, timeVal, lod, doCheap, volSample);
	vec3 emit = vec3(0.0);

	return CloudDensityResult(vec3(1.0) * res.density, advectSpeed, 1.0, vec3(1.0), emit, res.relativeExtinction);
}

#endif // HELPERS_CLOUDS_GLSL
