#version 460
#include "bindless.glsl"
#include "bindless_tlas.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform DeferredPushConstants {
	uvec4 gridParams; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
	uvec4 lodOffsets0_3;
	uvec4 lodOffsets4_7;
	uint  gPositionIndex;
	uint  gNormalIndex;
	uint  gAlbedoIndex;
	uint  backgroundIndex;
	uint  clipmapIndex;
	uint  tlasIndex;
} params;

vec2 sampleToroidalUV(vec2 worldXZ, uint level) {
	float baseTexelSize = (uCameraPosition.w > 0.0) ? uCameraPosition.w : 0.5;
	float texelSize = baseTexelSize * pow(2.0, float(level));
	uint textureDim = (params.gridParams.w > 0u) ? params.gridParams.w : 1088u;

	uvec4 offsets = (level < 4) ? params.lodOffsets0_3 : params.lodOffsets4_7;
	uint packed = offsets[level % 4];
	ivec2 gridOffset = ivec2(int(packed & 0xFFFFu), int((packed >> 16u) & 0xFFFFu));

	vec2 centerWorldPos = floor(uCameraPosition.xz / texelSize) * texelSize;
	vec2 deltaWorld = worldXZ - centerWorldPos;
	vec2 texelCoord = deltaWorld / texelSize + vec2(float(textureDim) * 0.5) + vec2(gridOffset);

	return fract(texelCoord / float(textureDim));
}

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

		vec2 uv = sampleToroidalUV(samplePos.xz, stepLod);
		// Sample the specific array layer matching the LOD
		vec4 texSample = SAMPLE_ARRAY_WRAP(params.clipmapIndex, vec3(uv, float(stepLod)));
		float terrainHeight = texSample.r;

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
		if (diff > 0.001) {
			vec3 rayOrigin = pos + norm * 0.1; // Base offset to avoid standard self-shadowing

			// Sample the absolute highest-detail terrain height at this coordinate
			vec2 uv0 = sampleToroidalUV(pos.xz, 0);
			float trueHeight0 = SAMPLE_ARRAY_WRAP(params.clipmapIndex, vec3(uv0, 0.0)).r;

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

		// Ambient term
		vec3 ambient = 0.25 * albedo.rgb;

		hdrColor = ambient + diffuse;
	}

	// HDR Tonemapping & Gamma Correction
	vec3 ldrColor = ACESFilm(hdrColor);
	ldrColor = pow(ldrColor, vec3(1.0 / 2.2));

	outColor = vec4(ldrColor, 1.0);
}
