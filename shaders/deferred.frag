#version 460
#include "bindless.glsl"
#include "bindless_tlas.glsl"
#include "common.glsl"
#include "terrain.glsl"
#include "clustered_lighting.glsl"

#define ATMOSPHERE_NO_PUSH_CONSTANTS
#include "atmosphere/common.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform DeferredPushConstants {
	uvec4 gridParams; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
	uvec4 lodOffsets0_3;
	uvec4 lodOffsets4_7;
	uvec4 lodOffsets8_11;
	uint  gPositionIndex;
	uint  gNormalIndex;
	uint  gAlbedoIndex;
	uint  backgroundIndex;
	uint  clipmapIndex;
	uint  tlasIndex;
	uint  gDepthIndex;
	uint  minMaxIndex;
	uint  biomeIndex;
	uint  visibilityIndex;
}

params;

// Calculate the LOD level based on the sample's Chebyshev distance
uint calculateRayLOD(vec2 sampleXZ) {
	vec2  dists = abs(sampleXZ - uCameraPosition.xz);
	float maxDist = max(dists.x, dists.y);

	float baseRadius = 272.0;

	if (maxDist < baseRadius) {
		return 0;
	}

	float lodFloat = ceil(log2(maxDist / baseRadius));
	return uint(clamp(lodFloat, 0.0, 7.0));
}

// Update the intersection function to use dynamic LODs
bool checkTerrainAABBIntersection(vec3 rayOrigin, vec3 rayDir, float camDistToShaded, out float hitT) {
	float stepSize = clamp(camDistToShaded * 0.01, 1.0, 5.0);
	int   numSteps = int(clamp(200.0 / stepSize, 25.0, 50.0));

	float rayLength = 500.0;
	for (int i = 1; i <= numSteps; ++i) {
		float t = (float(i) / float(numSteps)) * rayLength;
		vec3  samplePos = rayOrigin + rayDir * t;

		// Fetch the appropriate LOD for the current spatial step
		uint stepLod = calculateRayLOD(samplePos.xz);

		vec2  flatXZ = samplePos.xz - uCameraPosition.xz;
		float dropOff = dot(flatXZ, flatXZ) / (2.0 * FAKE_PLANET_RADIUS);

		// Accelerate raymarching via min-max height map check if minMaxIndex is set
		if (params.minMaxIndex > 0u) {
			vec2 minMax = sampleTerrainMinMax(
				params.minMaxIndex,
				samplePos.xz,
				stepLod,
				params.gridParams.w,
				params.lodOffsets0_3,
				params.lodOffsets4_7,
				params.lodOffsets8_11
			);
			if (samplePos.y > minMax.y - dropOff + 1.0) {
				continue; // Ray is safely above the maximum height in this cell
			}
		}

		vec4 texSample = sampleTerrainClipmap(
			params.clipmapIndex,
			samplePos.xz,
			stepLod,
			params.gridParams.w,
			params.lodOffsets0_3,
			params.lodOffsets4_7,
			params.lodOffsets8_11
		);
		float terrainHeight = texSample.r - dropOff;

		if (samplePos.y <= terrainHeight) {
			hitT = t;
			return true;
		}
	}
	hitT = 0.0;
	return false;
}

// ACES Filmic Tone Mapping Curve
vec3 ACESFilm(vec3 x) {
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
	vec4  albedo = SAMPLE_NEAREST(params.gAlbedoIndex, inUV);
	vec3  norm = SAMPLE_NEAREST(params.gNormalIndex, inUV).rgb;
	vec3  relPos = SAMPLE_NEAREST(params.gPositionIndex, inUV).rgb;
	vec3  pos = relPos + uCameraPosition.xyz;
	vec3  hdrBg = SAMPLE_NEAREST(params.backgroundIndex, inUV).rgb;
	float depth = SAMPLE_NEAREST(params.gDepthIndex, inUV).r;

	vec3 hdrColor;

	// If no surface rendered in gbuffer (albedo alpha is 0), show HDR background
	if (albedo.a < 0.01) {
		hdrColor = hdrBg;
	} else {
		float shadowFactor = 1.0;

		vec3 lightContribution = evaluateClusteredLightContribution(pos, norm);
		vec3 diffuse = albedo.rgb * lightContribution * shadowFactor;

		// Extract primary directional light for aerial perspective / atmosphere scattering
		vec3 sunDir = normalize(vec3(0.4, 0.8, 0.4));
		vec3 sunRadiance = vec3(10.0, 9.5, 8.5);

		uint numDirectionalCheck = min(uLightCount, 16u);
		for (uint i = 0u; i < numDirectionalCheck; ++i) {
			if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL) {
				sunDir = normalize(uLights[i].direction);
				sunRadiance = uLights[i].color * uLights[i].intensity;
				break;
			}
		}

		float distMeters = length(relPos);
		float distKM = distMeters / 1000.0;
		vec3  rayDir = relPos / max(0.001, distMeters);

		// Evaluate atmospheric scattering, aerial perspective, and exponential fog
		vec3 transmittance = vec3(1.0);
		vec3 rayCamOriginKM = uCameraPosition.xyz / 1000.0;
		vec3 inScattered = evaluateAerialPerspective(rayCamOriginKM, rayDir, distKM, sunDir, sunRadiance, transmittance);

		// Scaffolding: Volumetric lighting along view ray to surface
		vec3 vLight = evaluateVolumetricLighting(rayCamOriginKM, rayDir, distKM, sunDir, sunRadiance);

		hdrColor = (diffuse * vLight) * transmittance + inScattered;

		// Underwater Medium Adaptation for rendered surfaces
		if (uCameraPosition.y < u_waterLevel) {
			float depthBelowWater = (u_waterLevel - uCameraPosition.y);
			float rayWaterLength = min(distMeters, depthBelowWater / max(0.01, abs(rayDir.y)));
			vec3  waterTransmittance = exp(-kWaterExtinction * u_waterScale * (rayWaterLength / 1000.0));
			vec3  waterFogColor = kWaterScattering * u_waterScale * vec3(0.12, 0.62, 0.78);
			hdrColor = mix(waterFogColor, hdrColor * waterTransmittance, clamp(exp(-rayWaterLength * 0.01), 0.0, 1.0));
		}
	}

	// HDR Tonemapping & Gamma Correction
	vec3 ldrColor = ACESFilm(hdrColor);
	ldrColor = pow(ldrColor, vec3(1.0 / 2.2));

	outColor = vec4(ldrColor, 1.0);
}
