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
	uint  weatherBiomeIndex;
	uint  gMaterialIndex;
	uint  skyViewIndex;
	uint  gtaoIndex;
	float aoDirectInfluence;
	uint  debugVisualizeGTAO;
	uint  cloudShadowIndex;
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
	vec4  posSample = SAMPLE_NEAREST(params.gPositionIndex, inUV);
	vec3  relPos = posSample.rgb;
	vec3  pos = relPos + uCameraPosition.xyz;
	vec3  hdrBg = SAMPLE_NEAREST(params.backgroundIndex, inUV).rgb;
	float depth = SAMPLE_NEAREST(params.gDepthIndex, inUV).r;

	vec3 hdrColor;

	// If no surface rendered in gbuffer (albedo alpha is 0), show HDR background
	if (albedo.a < 0.01) {
		hdrColor = hdrBg;
	} else {
		vec4 matSample = SAMPLE_NEAREST(params.gMaterialIndex, inUV);

		float metallic   = matSample.r;
		float roughness  = matSample.g > 0.0 ? matSample.g : 0.7;
		float glint      = matSample.b;
		float ao         = normalSample.a > 0.0 ? normalSample.a : 1.0;
		float emissivity = posSample.a;

		if (params.debugVisualizeGTAO != 0u) {
			float sampledGTAO = params.gtaoIndex > 0u ? SAMPLE_NEAREST(params.gtaoIndex, inUV).r : 1.0;
			outColor = vec4(vec3(sampledGTAO), 1.0);
			return;
		}

		Material material = Material(albedo.rgb, roughness, metallic, ao, emissivity, glint);

		// Aerial perspective / underwater extinction is no longer applied here: it happens
		// uniformly for every pixel (this one included) in AtmosphereCompositeNode, which runs
		// after this pass at SubPhase::Atmosphere -- see shaders/atmosphere/composite.frag.
		hdrColor = evaluateClusteredLightContributionPBR(
			pos,
			norm,
			material,
			params.skyViewIndex,
			params.weatherBiomeIndex,
			params.gtaoIndex,
			inUV,
			params.aoDirectInfluence
		).color;

		if (uLights[0].intensity > 0.0) {
			TerrainOcclusionContext occCtx = makeTerrainOcclusionContext(
				params.minMaxIndex,
				params.gridParams.w,
				params.gridParams.x,
				0.0
			);

			vec3 lightDir = normalize(-uLights[0].direction);

			// Sample smooth geometric terrain clipmap normal for coherent origin biasing and light angle evaluation
			vec3 geoNorm = norm;
			// if (params.clipmapIndex > 0u && params.clipmapIndex != 0xFFFFFFFFu) {
			// 	vec4 clipmapSample = sampleTerrainClipmap(params.clipmapIndex, pos.xz, 0u, params.gridParams.w);
			// 	if (length(clipmapSample.gba) >= 0.1) {
			// 		geoNorm = clipmapSample.gba;
			// 	}
			// }

			float NdotL = dot(geoNorm, lightDir);
			float shadowMask = 1.0;

			// Soft shadow penumbra sharpness factor
			const float kPenumbra = 8.0;

			// Soft grazing-angle transition factor matching the penumbra soft shadow curve
			float softNdotL = clamp(NdotL * kPenumbra, 0.0, 1.0);

			if (softNdotL <= 0.0) {
				// Fast exit for surfaces facing away from light or at extreme grazing angles
				shadowMask = 0.0;
			} else {
				vec3 biasNorm = norm;//normalize(cross(dFdx(pos), dFdy(pos)));
				float cosTheta = clamp(dot(biasNorm, lightDir), 0.0, 1.0);
				float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
				float minBias = 0.1;
				float maxBias = 0.5;
				vec3 biasVector = (lightDir * cosTheta * minBias) + (biasNorm * sinTheta * maxBias);

				vec3 rayOrigin = pos + biasVector;

				float raySoftShadow = calculateTerrainSoftShadow(occCtx, rayOrigin, lightDir, 20000.0, kPenumbra);

				// Combine ray soft shadow with facing-away grazing angle transition for continuous C0 soft shadow
				shadowMask = min(raySoftShadow, softNdotL);
			}

			if (params.cloudShadowIndex > 0u) {
				vec2 projXZ = pos.xz - (pos.y / max(lightDir.y, 0.001)) * lightDir.xz;
				vec2 shadowUV = (projXZ - uCameraPosition.xz) / 160000.0 + 0.5;
				if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0) {
					vec4 cloudShadowData = SAMPLE_ARRAY_CLAMP(params.cloudShadowIndex, vec3(shadowUV, 4.0));
					shadowMask *= cloudShadowData.a;
				}
			}

			float shadowStrength = 0.5*clamp(uLights[0].intensity, 0.0, 1.0);
			hdrColor *= mix(1.0, shadowMask, shadowStrength);
		}
	}

	outColor = vec4(hdrColor, 1.0);
}
