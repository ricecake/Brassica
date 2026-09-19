#ifndef BRASSICA_CLUSTERED_LIGHTING_GLSL
#define BRASSICA_CLUSTERED_LIGHTING_GLSL

#include "lighting.glsl"
#include "helpers/lighting.glsl"

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
vec3 evaluateClusteredLightContributionPBR(vec3 frag_pos, vec3 normal, vec3 albedo, float roughness, float metallic, float ao) {
	vec3 N = normalize(normal);
	vec3 V = normalize(uCameraPosition.xyz - frag_pos);

	vec3 F0 = mix(vec3(0.04), albedo, metallic);
	vec3 Lo = vec3(0.0);
	float spec_lum = 0.0;

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
			evaluate_brdf(N, V, L, albedo, roughness, metallic, F0, radiance, 1.0, Lo, spec_lum);
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
		evaluate_brdf(N, V, L, albedo, roughness, metallic, F0, radiance, 1.0, Lo, spec_lum);
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
		evaluate_brdf(N, V, L, albedo, roughness, metallic, F0, radiance, 1.0, Lo, spec_lum);
	}

	float terrainOcc = calculateTerrainOcclusion(frag_pos, N);
	vec3 spatialSHAmbient = getSpatialAmbientSH(frag_pos, N);
	vec3 ambient = spatialSHAmbient * uAmbientLight.rgb * albedo * (ao * terrainOcc);

	return ambient + Lo;
}

/**
 * Standard diffuse/specular wrapper for existing shaders.
 */
vec3 evaluateClusteredLightContribution(vec3 frag_pos, vec3 normal) {
	return evaluateClusteredLightContributionPBR(frag_pos, normal, vec3(1.0), 0.7, 0.0, 1.0);
}

/**
 * Helper to evaluate clustered local light contribution without normals.
 */
vec3 evaluateClusteredLightContributionSimple(vec3 frag_pos) {
	return evaluateClusteredLightContributionPBR(frag_pos, vec3(0.0, 1.0, 0.0), vec3(1.0), 1.0, 0.0, 1.0);
}

#endif // BRASSICA_CLUSTERED_LIGHTING_GLSL
