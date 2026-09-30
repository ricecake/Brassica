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

// Add this helper for shadow jitter
float interleavedGradientNoise(vec2 screenPos) {
	vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
	return fract(magic.z * fract(dot(screenPos, magic.xy)));
}

// Updated intersection function
bool checkTerrainAABBIntersection(vec3 rayOrigin, vec3 rayDir, float camDistToShaded, out float hitT) {
	// return true;
	float stepSize = clamp(camDistToShaded * 0.01, 0.1250, 15.0);
	int   numSteps = int(clamp(200.0 / stepSize, 5.0, 150.0));
	float rayLength = 1000.0; // Consider clamping this dynamically based on atmosphere ceiling

	for (int i = 1; i <= numSteps; ++i) {
		float t = (float(i) / float(numSteps)) * rayLength;
		vec3  samplePos = rayOrigin + rayDir * t;

		uint stepLod = 2+calculateRayLOD(samplePos.xz);

		vec2  flatXZ = samplePos.xz - uCameraPosition.xz;
		float dropOff = dot(flatXZ, flatXZ) / (2.0 * FAKE_PLANET_RADIUS);

		// Adjust the test altitude by the curvature dropoff
		float testAlt = samplePos.y + dropOff;

		if (params.minMaxIndex > 0u) {
			vec2 minMax = sampleTerrainMinMax(
				params.minMaxIndex,
				samplePos.xz,
				stepLod,
				params.gridParams.w
			);

			// minMax.g (or .y) contains the cluster_max from your terrain_gen pass
			if (testAlt > minMax.y-1.0) {
				continue;
			}
		}

		vec4 texSample = sampleTerrainClipmap(
			params.clipmapIndex,
			samplePos.xz,
			stepLod,
			params.gridParams.w
		);

		if (testAlt <= texSample.r) {
			hitT = t;
			return true;
		}
	}
	hitT = 0.0;
	return false;
}

float rayQueryShadow(vec3 pos, vec3 norm) {
	vec3 lightDir = -uLights[0].direction;

	// Early out for rays originating high up and pointing into the sky
	if (pos.y > 2500.0 && lightDir.y > 0.0) {
		return 1.0;
	}

	// 1. Normal-scaled bias replaces the expensive LOD0 clipmap fetch
	float biasAmount = 0.15;
	vec3 rayOrigin = pos + (norm * biasAmount);

	// 2. Add IGN jitter to tMin to break up Moire banding across LOD morphs
	float noise = interleavedGradientNoise(gl_FragCoord.xy);
	float tMin = 0.05 + (noise * 0.2);

	float shadowRayTMax = 10000.0;

	// 3. Add TerminateOnFirstHitEXT and SkipClosestHitShaderEXT
	uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT;

	rayQueryEXT rq;
	rayQueryInitializeEXT(
		rq,
		uTLAS[nonuniformEXT(params.tlasIndex)],
		rayFlags,
		0xFF,
		rayOrigin,
		tMin,
		lightDir,
		shadowRayTMax
	);

	float camDistToShaded = length(pos);

	while (rayQueryProceedEXT(rq)) {
		uint candidateType = rayQueryGetIntersectionTypeEXT(rq, false);
		if (candidateType == gl_RayQueryCandidateIntersectionAABBEXT) {
			float hitT;
			if (checkTerrainAABBIntersection(rayOrigin, lightDir, camDistToShaded, hitT)) {
				// Because gl_RayFlagsTerminateOnFirstHitEXT is active,
				// this instantly commits the hit and terminates the while loop.
				rayQueryGenerateIntersectionEXT(rq, hitT);
			}
		}
	}

	float shadowFactor = 1.0;
	if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
		shadowFactor = 0.2;
	}
	return shadowFactor;
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
		Material material = Material(albedo.rgb, roughness, 0.0, 1.0);

		float shadowFactor = rayQueryShadow(pos, norm);

		hdrColor = evaluateClusteredLightContributionPBR(pos, norm, material, shadowFactor).color;
	}

	outColor = vec4(hdrColor, 1.0);
}
