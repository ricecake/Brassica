#ifndef BRASSICA_CLUSTERED_LIGHTING_GLSL
#define BRASSICA_CLUSTERED_LIGHTING_GLSL

#include "lighting.glsl"
#include "helpers/lighting.glsl"
#include "helpers/octahedral.glsl"
#include "atmosphere/common.glsl"
#include "helpers/whittaker.glsl"
#include "external/lygia/lighting/common/gtaoMultiBounce.glsl"

/**
 * Computes the 1D cluster index for a given world-space position.
 */
uint getClusterIndex(vec3 frag_pos) {
	vec4 view_pos = uViewMatrix * vec4(frag_pos, 1.0);
	float z_val = -view_pos.z;

	float z_near = uNearPlane;
	float z_far = uFarPlane;

	int z_slice = int(clamp(log(max(z_val, 0.001) / z_near) / log(z_far / z_near) * 24.0, 0.0, 23.0));

	vec4 clip_pos = uProjMatrix * view_pos;
	vec3 ndc_pos = clip_pos.xyz / max(0.0001, clip_pos.w);
	vec2 screen_uv = ndc_pos.xy * 0.5 + 0.5;

	int x_slice = int(clamp(screen_uv.x * 16.0, 0.0, 15.0));
	int y_slice = int(clamp(screen_uv.y * 9.0, 0.0, 8.0));

	return uint(x_slice + y_slice * 16 + z_slice * 16 * 9);
}

/**
 * High-level GLSL helper to evaluate clustered local light contribution with Cook-Torrance PBR BRDF.
 */
LightingResult evaluateClusteredLightContributionPBR(
	vec3 frag_pos,
	vec3 normal,
	Material material,
	uint skyViewIndex,
	uint weatherBiomeIndex,
	uint gtaoIndex,
	vec2 uv
) {
	vec3 N = normalize(normal);
	vec3 V = normalize(uCameraPosition.xyz - frag_pos);

	LightingResult result = LightingResult(vec3(0.0), 0.0);

	// Evaluate directional lights (indices 0 and 1)
	uint dir_count = min(uLightCount, 2u);
	for (uint i = 0u; i < dir_count; ++i) {
		if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL && uLights[i].intensity > 0.0) {
			vec3 L;
			float attenuation;
			calculateLightContribution(
				uLights[i].type,
				uLights[i].position,
				uLights[i].direction,
				uLights[i].innerCutoff,
				uLights[i].outerCutoff,
				uLights[i].intensity,
				frag_pos,
				L,
				attenuation
			);

			vec3 radiance = uLights[i].color * (uLights[i].intensity * PBR_INTENSITY_BOOST) * attenuation;
			evaluate_brdf(N, V, L, material, radiance, 1.0, result);
		}
	}

	// Evaluate global lights bucket (index 3456)
	Cluster global_cluster = uClusters[3456];
	for (uint i = 0; i < global_cluster.count; ++i) {
		uint light_index = global_cluster.lightIndices[i];
		if (uLights[light_index].intensity <= 0.0) continue;

		vec3 light_pos = uLights[light_index].position;
		if ((uLights[light_index].flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			uLights[light_index].type,
			light_pos,
			uLights[light_index].direction,
			uLights[light_index].innerCutoff,
			uLights[light_index].outerCutoff,
			uLights[light_index].intensity,
			frag_pos,
			L,
			attenuation
		);

		if (attenuation <= 0.0) continue;

		vec3 radiance = uLights[light_index].color * (uLights[light_index].intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}

	uint cluster_index = getClusterIndex(frag_pos);
	Cluster cluster = uClusters[cluster_index];

	for (uint i = 0; i < cluster.count; ++i) {
		uint light_index = cluster.lightIndices[i];
		if (uLights[light_index].intensity <= 0.0) continue;

		vec3 light_pos = uLights[light_index].position;
		if ((uLights[light_index].flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			uLights[light_index].type,
			light_pos,
			uLights[light_index].direction,
			uLights[light_index].innerCutoff,
			uLights[light_index].outerCutoff,
			uLights[light_index].intensity,
			frag_pos,
			L,
			attenuation
		);

		if (attenuation <= 0.0) continue;

		vec3 radiance = uLights[light_index].color * (uLights[light_index].intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}

	float terrainOcc = calculateTerrainOcclusion(frag_pos, N);

	// 1. Minimum physically grounded airglow floor (upper atmosphere chemiluminescence)
	const vec3 airglow = vec3(0.0008, 0.0015, 0.0022);

	// 2. Sky Light from SkyViewLUT
	vec3 skyLight = airglow;
	if (skyViewIndex > 0u) {
		vec3 skyZenith = sampleSkyView(skyViewIndex, vec3(0.0, 1.0, 0.0));
		vec3 skyNorm = sampleSkyView(skyViewIndex, max(N, vec3(0.0, 0.01, 0.0)));
		skyLight = mix(skyZenith, skyNorm, 0.5);
	}

	// 3. Cloud coverage & Weather modulation
	vec3 biomeColor = vec3(0.2, 0.35, 0.15);
	if (weatherBiomeIndex > 0u) {
		vec3 planetDir = normalize(frag_pos - vec3(0.0, -FAKE_PLANET_RADIUS, 0.0));
		vec2 octUV = directionToOctahedralUV(planetDir);
		vec4 weather = SAMPLE_LINEAR(weatherBiomeIndex, octUV);

		float cloudCoverage = clamp(weather.r, 0.0, 1.0);
		float cloudAttenuation = mix(1.0, 0.35, cloudCoverage);
		vec3 overcastSky = vec3(length(skyLight) * 0.6);
		skyLight = mix(skyLight, overcastSky, cloudCoverage * 0.7) * cloudAttenuation;

		WhittakerBiome biome = evaluateWhittakerBiome(weather.y, weather.z, weather.a, weather.g);
		biomeColor = biome.color;
	}

	skyLight = max(skyLight, airglow);

	// 4. Biome Palette ground bounce (hemispheric upward ambient)
	vec3 groundBounce = skyLight * biomeColor * 0.4;
	float hemisphere = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
	vec3 ambientLight = mix(groundBounce, skyLight, hemisphere);

	// 5. GTAO Modulation with multi-bounce inter-reflection
	float gtaoVal = material.ao;
	if (gtaoIndex > 0u) {
		float sampledGTAO = SAMPLE_NEAREST(gtaoIndex, uv).r;
		gtaoVal *= sampledGTAO;
	}

	vec3 multiBounceAO = gtaoMultiBounce(gtaoVal, material.albedo);

	vec3 spatialSHAmbient = getSpatialAmbientSH(frag_pos, N);
	vec3 finalAmbient = (spatialSHAmbient + ambientLight) * uAmbientLight.rgb;

	result.color += finalAmbient * material.albedo * multiBounceAO * terrainOcc;
	result.color += material.albedo * material.emissivity;

	return result;
}

LightingResult evaluateClusteredLightContributionPBR(vec3 frag_pos, vec3 normal, Material material) {
	return evaluateClusteredLightContributionPBR(frag_pos, normal, material, 0u, 0u, 0u, vec2(0.5));
}

#endif // BRASSICA_CLUSTERED_LIGHTING_GLSL
