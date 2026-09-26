#version 460
#include "bindless.glsl"
#include "bindless_tlas.glsl"
#include "common.glsl"
#include "terrain.glsl"
#include "clustered_lighting.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform DeferredPushConstants {
	uvec4 gridParams; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
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
	uint  weatherBiomeIndex;
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

		uint stepLod = calculateRayLOD(samplePos.xz);

		vec2  flatXZ = samplePos.xz - uCameraPosition.xz;
		float dropOff = dot(flatXZ, flatXZ) / (2.0 * FAKE_PLANET_RADIUS);

		if (params.minMaxIndex > 0u) {
			vec2 minMax = sampleTerrainMinMax(
				params.minMaxIndex,
				samplePos.xz,
				stepLod,
				params.gridParams.w
			);
			if (samplePos.y > minMax.y - dropOff + 1.0) {
				continue;
			}
		}

		vec4 texSample = sampleTerrainClipmap(
			params.clipmapIndex,
			samplePos.xz,
			stepLod,
			params.gridParams.w
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

void main() {
	vec4  albedo = SAMPLE_NEAREST(params.gAlbedoIndex, inUV);
	vec4  normalSample = SAMPLE_NEAREST(params.gNormalIndex, inUV);
	vec3  norm = normalSample.rgb;
	vec3  relPos = SAMPLE_NEAREST(params.gPositionIndex, inUV).rgb;
	vec3  pos = relPos + uCameraPosition.xyz;
	vec3  hdrBg = SAMPLE_NEAREST(params.backgroundIndex, inUV).rgb;
	float depth = SAMPLE_NEAREST(params.gDepthIndex, inUV).r;

	vec3 hdrColor;

	// If no surface rendered in gbuffer (albedo alpha is 0), show HDR background
	if (albedo.a < 0.01) {
		hdrColor = hdrBg;
	} else {
		float roughness = normalSample.a > 0.0 ? normalSample.a : 0.7;

		if (params.weatherBiomeIndex > 0u) {
			vec4 weatherSample = sampleTerrainWeatherBiome(params.weatherBiomeIndex, pos);
			float severity = weatherSample.g;
			float rainShadow = weatherSample.b;
			float snowCover = weatherSample.a;
			float tempEstimate = clamp(1.0 - (pos.y + 100.0) / 1500.0, 0.0, 1.0);
			WhittakerBiome wb = evaluateWhittakerBiome(tempEstimate, clamp(0.5 - rainShadow * 0.3, 0.0, 1.0), severity, snowCover > 0.1 ? 1.0 : 0.0);
			albedo.rgb = mix(albedo.rgb, wb.color, 0.65);
			roughness = mix(roughness, wb.roughness, 0.65);
		}

		Material material = Material(albedo.rgb, roughness, 0.0, 1.0);

		// Aerial perspective / underwater extinction is no longer applied here: it happens
		// uniformly for every pixel (this one included) in AtmosphereCompositeNode, which runs
		// after this pass at SubPhase::Atmosphere -- see shaders/atmosphere/composite.frag.
		hdrColor = evaluateClusteredLightContributionPBR(pos, norm, material).color;
	}

	outColor = vec4(hdrColor, 1.0);
}
