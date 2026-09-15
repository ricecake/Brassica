#version 460
#include "bindless.glsl"
#include "bindless_tlas.glsl"
#include "common.glsl"
#include "terrain.glsl"

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
	uint gDepthIndex;
	uint  minMaxIndex;
	uint  biomeIndex;
	uint  visibilityIndex;
} params;

// Calculate the LOD level based on the sample's Chebyshev distance
uint calculateRayLOD(vec2 sampleXZ) {
	vec2 dists = abs(sampleXZ - uCameraPosition.xz);
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
	int numSteps = int(clamp(200.0 / stepSize, 25.0, 50.0));

	float rayLength = 500.0;
	for (int i = 1; i <= numSteps; ++i) {
		float t = (float(i) / float(numSteps)) * rayLength;
		vec3 samplePos = rayOrigin + rayDir * t;

		// Fetch the appropriate LOD for the current spatial step
		uint stepLod = calculateRayLOD(samplePos.xz);

		vec2 flatXZ = samplePos.xz - uCameraPosition.xz;
		float dropOff = dot(flatXZ, flatXZ) / (2.0 * FAKE_PLANET_RADIUS);

		// Accelerate raymarching via min-max height map check if minMaxIndex is set
		if (params.minMaxIndex > 0u) {
			vec2 minMax = sampleTerrainMinMax(params.minMaxIndex, samplePos.xz, stepLod, params.gridParams.w, params.lodOffsets0_3, params.lodOffsets4_7, params.lodOffsets8_11);
			if (samplePos.y > minMax.y - dropOff + 1.0) {
				continue; // Ray is safely above the maximum height in this cell
			}
		}

		vec4 texSample = sampleTerrainClipmap(params.clipmapIndex, samplePos.xz, stepLod, params.gridParams.w, params.lodOffsets0_3, params.lodOffsets4_7, params.lodOffsets8_11);
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
	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, inUV);
	vec3 norm = SAMPLE_NEAREST(params.gNormalIndex, inUV).rgb;
	vec3 pos = SAMPLE_NEAREST(params.gPositionIndex, inUV).rgb;
	vec3 hdrBg = SAMPLE_NEAREST(params.backgroundIndex, inUV).rgb;
	float depth = SAMPLE_NEAREST(params.gDepthIndex, inUV).r;

	vec3 hdrColor;

	// If no surface rendered in gbuffer (albedo alpha is 0), show HDR background
	if (albedo.a < 0.01) {
		hdrColor = hdrBg;
	} else {
		vec3 lightDir = normalize(vec3(0.5, 0.2, 0.5));
		vec3 lightColor = vec3(2.5, 2.3, 2.0); // High intensity HDR light source

		float diff = max(dot(norm, lightDir), 0.0);

		// Ray Query Shadows
		float shadowFactor = 1.0;

		// Inside main(), replace the existing rayOrigin assignment:
		if (false && diff > 0.001) {
			vec3 rayOrigin = pos + norm * 0.1; // Base offset to avoid standard self-shadowing

			// Sample the absolute highest-detail terrain height at this coordinate
			float trueHeight0 = sampleTerrainClipmap(params.clipmapIndex, pos.xz, 0u, params.gridParams.w, params.lodOffsets0_3, params.lodOffsets4_7, params.lodOffsets8_11).r;

			// Dynamically push the ray origin above the LOD 0 surface if the geometry is buried
			if (rayOrigin.y < trueHeight0 + 0.1) {
				rayOrigin.y = trueHeight0 + 0.1;
			}

			float shadowRayTMax = 1000.0;


			rayQueryEXT rq;
			rayQueryInitializeEXT(
				rq,
				uTLAS[nonuniformEXT(params.tlasIndex)],
				gl_RayFlagsNoneEXT,
				0xFF,
				rayOrigin,
				0.1,
				lightDir,
				shadowRayTMax
			);

			float camDistToShaded = length(pos);

			while (rayQueryProceedEXT(rq)) {
				uint candidateType = rayQueryGetIntersectionTypeEXT(rq, false);
				if (candidateType == gl_RayQueryCandidateIntersectionAABBEXT) {
					float hitT;
					if (checkTerrainAABBIntersection(rayOrigin, lightDir, camDistToShaded, hitT)) {
						rayQueryGenerateIntersectionEXT(rq, hitT);
					}
				}
			}

			if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
				shadowFactor = 0.2; // Shadowed region
			}
		}

		vec3 diffuse = albedo.rgb * diff * lightColor * shadowFactor;
		// diffuse = mix(vec3(0.1,0.2, 0.3), diffuse, exp(-0.10*length(pos)));
		// diffuse += vec3(1.2, 0.2, 0.2) * exp(-length(pos));

		// Ambient term
		vec3 ambient = 0.25 * albedo.rgb;

		hdrColor = ambient + diffuse;
	}

	// HDR Tonemapping & Gamma Correction
	vec3 ldrColor = ACESFilm(hdrColor);
	ldrColor = pow(ldrColor, vec3(1.0 / 2.2));

	outColor = vec4(ldrColor, 1.0);
}
