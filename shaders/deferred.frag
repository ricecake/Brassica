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

vec2 sampleToroidalUV(vec2 worldXZ, uint level) {
	float baseTexelSize = 0.5;
	float texelSize = baseTexelSize * pow(2.0, float(level));
	float worldExtent = 1024.0 * texelSize;

	vec2 linearUV = (worldXZ + vec2(worldExtent * 0.5)) / worldExtent;
	return fract(linearUV);
}

// Ray query AABB candidate heightmap traversal
bool checkTerrainAABBIntersection(vec3 rayOrigin, vec3 rayDir, float camDistToShaded, out float hitT) {
	// Traversal precision determined by distance from camera to the shaded point (ray origin).
	// Closer shaded points get higher precision sampling steps along the ray within the AABB.
	float stepSize = clamp(camDistToShaded * 0.01, 0.5, 4.0);
	int numSteps = int(clamp(200.0 / stepSize, 10.0, 50.0));

	float rayLength = 500.0;
	for (int i = 1; i <= numSteps; ++i) {
		float t = (float(i) / float(numSteps)) * rayLength;
		vec3 samplePos = rayOrigin + rayDir * t;

		vec2 uv = sampleToroidalUV(samplePos.xz, 0);
		vec4 texSample = texture(terrainClipmap, vec3(uv, 0.0));
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
		vec3 lightDir = normalize(vec3(sin(ubo.time), 0.8, -cos(ubo.time)));
		vec3 lightColor = vec3(2.5, 2.3, 2.0); // High intensity HDR light source

		float diff = max(dot(norm, lightDir), 0.0);

		// Ray Query Shadows
		float shadowFactor = 1.0;
		if (diff > 0.001) {
			vec3 rayOrigin = pos + norm * 0.1; // Offset to avoid self-shadowing
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
