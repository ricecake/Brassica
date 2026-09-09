#version 460
#extension GL_EXT_ray_query : enable

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	uint  frameIndex;
	uint  globalSeed;
	uint  frameRandom;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D gPosition;
layout(set = 1, binding = 1) uniform sampler2D gNormal;
layout(set = 1, binding = 2) uniform sampler2D gAlbedo;
layout(set = 1, binding = 3) uniform sampler2D backgroundTex;
layout(set = 1, binding = 4) uniform sampler2DArray terrainClipmap;
layout(set = 1, binding = 5) uniform accelerationStructureEXT topLevelAS;

struct TerrainAABB {
	float minX;
	float minY;
	float minZ;
	float maxX;
	float maxY;
	float maxZ;
};

layout(std430, set = 1, binding = 6) readonly buffer TerrainAABBs {
	TerrainAABB terrainAABBs[];
};

layout(set = 1, binding = 7) uniform sampler3D volumetricIntegratedTex;

layout(push_constant) uniform TerrainPushConstants {
	mat4  viewProj;
	vec4  cameraPos;
	uvec4 gridParams;
	uvec4 lodOffsets0_3;
	uvec4 lodOffsets4_7;
} params;

vec2 sampleToroidalUV(vec2 worldXZ, uint level) {
	float baseTexelSize = (params.cameraPos.w > 0.0) ? params.cameraPos.w : 0.5;
	float texelSize = baseTexelSize * pow(2.0, float(level));
	uint textureDim = (params.gridParams.w > 0u) ? params.gridParams.w : 1088u;

	uvec4 offsets = (level < 4) ? params.lodOffsets0_3 : params.lodOffsets4_7;
	uint packed = offsets[level % 4];
	ivec2 gridOffset = ivec2(int(packed & 0xFFFFu), int((packed >> 16u) & 0xFFFFu));

	vec2 centerWorldPos = floor(params.cameraPos.xz / texelSize) * texelSize;
	vec2 deltaWorld = worldXZ - centerWorldPos;
	vec2 texelCoord = deltaWorld / texelSize + vec2(float(textureDim) * 0.5) + vec2(gridOffset);

	return fract(texelCoord / float(textureDim));
}

// Calculate the LOD level based on the sample's Chebyshev distance
uint calculateRayLOD(vec2 sampleXZ) {
	vec2 dists = abs(sampleXZ - params.cameraPos.xz);
	float maxDist = max(dists.x, dists.y);

	float baseRadius = 272.0;

	if (maxDist < baseRadius) {
		return 0;
	}

	float lodFloat = ceil(log2(maxDist / baseRadius));
	return uint(clamp(lodFloat, 0.0, 7.0));
}

// Heightmap raymarching constrained strictly within bounding box entry and exit parameters
bool checkTerrainAABBIntersectionInRange(vec3 rayOrigin, vec3 rayDir, float tMin, float tMax, out float hitT) {
	if (tMax <= tMin) {
		hitT = 0.0;
		return false;
	}
	float range = tMax - tMin;
	int numSteps = int(clamp(range * 0.1, 10.0, 40.0));

	for (int i = 0; i <= numSteps; ++i) {
		float t = tMin + (float(i) / float(numSteps)) * range;
		vec3 samplePos = rayOrigin + rayDir * t;

		// Fetch the appropriate LOD for the current spatial step
		uint stepLod = calculateRayLOD(samplePos.xz);

		vec2 uv = sampleToroidalUV(samplePos.xz, stepLod);
		// Sample the specific array layer matching the LOD
		vec4 texSample = texture(terrainClipmap, vec3(uv, float(stepLod)));
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
	vec4 albedo = texture(gAlbedo, inUV);
	vec3 norm = texture(gNormal, inUV).rgb;
	vec3 pos = texture(gPosition, inUV).rgb;
	vec3 hdrBg = texture(backgroundTex, inUV).rgb;

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
			float trueHeight0 = texture(terrainClipmap, vec3(uv0, 0.0)).r;

			// Dynamically push the ray origin above the LOD 0 surface if the geometry is buried
			if (rayOrigin.y < trueHeight0 + 0.1) {
				rayOrigin.y = trueHeight0 + 0.1;
			}

			float shadowRayTMax = 1000.0;


			rayQueryEXT rq;
			rayQueryInitializeEXT(
				rq,
				topLevelAS,
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
					uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
					TerrainAABB box = terrainAABBs[primIdx];
					vec3 bMin = vec3(box.minX, box.minY, box.minZ);
					vec3 bMax = vec3(box.maxX, box.maxY, box.maxZ);

					vec3 invDir = 1.0 / lightDir;
					vec3 t0 = (bMin - rayOrigin) * invDir;
					vec3 t1 = (bMax - rayOrigin) * invDir;
					vec3 tNear = min(t0, t1);
					vec3 tFar = max(t0, t1);
					float tMin = max(max(tNear.x, tNear.y), tNear.z);
					float tMax = min(min(tFar.x, tFar.y), tFar.z);

					float hitT;
					if (checkTerrainAABBIntersectionInRange(rayOrigin, lightDir, max(0.0, tMin), tMax, hitT)) {
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

	// Volumetric Fog Integration across 4 Cascades
	float viewDepth = length(pos - params.cameraPos.xyz);
	if (albedo.a < 0.01) {
		viewDepth = 1000.0; // Distant background
	}

	float cascadeDists[4] = float[4](20.0, 60.0, 200.0, 1000.0);
	int cascade = -1;
	float z_near = 0.1;
	float z_far = 1000.0;

	for (int i = 0; i < 4; ++i) {
		if (viewDepth <= cascadeDists[i]) {
			cascade = i;
			z_far = cascadeDists[i];
			if (i > 0) z_near = cascadeDists[i - 1];
			break;
		}
	}

	float w = 1.0;
	if (cascade != -1) {
		float slice = clamp(log(max(viewDepth, z_near) / z_near) / log(z_far / z_near), 0.0, 1.0);
		w = (float(cascade) + slice) / 4.0;
	}

	vec4 volSample = texture(volumetricIntegratedTex, vec3(inUV, w));
	hdrColor = hdrColor * volSample.a + volSample.rgb;

	// HDR Tonemapping & Gamma Correction
	vec3 ldrColor = ACESFilm(hdrColor);
	ldrColor = pow(ldrColor, vec3(1.0 / 2.2));

	outColor = vec4(ldrColor, 1.0);
}
