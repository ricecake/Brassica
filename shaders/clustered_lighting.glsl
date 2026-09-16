#ifndef BRASSICA_CLUSTERED_LIGHTING_GLSL
#define BRASSICA_CLUSTERED_LIGHTING_GLSL

#include "lighting.glsl"

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
 * High-level GLSL helper to evaluate clustered local light contribution with normal.
 */
vec3 evaluateClusteredLightContribution(vec3 frag_pos, vec3 normal) {
	vec3 n = normalize(normal);
	vec3 total_light = uAmbientLight.rgb;

	float exposureScale = 1.0;

	// Evaluate directional lights (indices 0 and 1)
	uint dir_count = min(uLightCount, 2u);
	for (uint i = 0u; i < dir_count; ++i) {
		if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL && uLights[i].intensity > 0.0) {
			vec3 L = normalize(-uLights[i].direction);
			float NdotL = max(dot(n, L), 0.0);
			total_light += uLights[i].color * uLights[i].intensity * NdotL;
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
		float attenuation = 1.0;

		if (uLights[light_index].type == LIGHT_TYPE_POINT) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_SPOT) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));

			float theta = dot(L, normalize(-uLights[light_index].direction));
			float epsilon = uLights[light_index].innerCutoff - uLights[light_index].outerCutoff;
			float angular_intensity = clamp((theta - uLights[light_index].outerCutoff) / epsilon, 0.0, 1.0);
			attenuation *= angular_intensity;
		} else if (uLights[light_index].type == LIGHT_TYPE_EMISSIVE) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			float emissive_radius = uLights[light_index].innerCutoff;
			float effective_dist = max(distance - emissive_radius * 0.5, 0.0);
			attenuation = 1.0 / (1.0 + 0.09 * effective_dist + 0.032 * effective_dist * effective_dist);
			float proximity_boost = smoothstep(emissive_radius * 2.0, 0.0, distance);
			attenuation = mix(attenuation, 1.0, proximity_boost * 0.5);
			float radius = (sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 + emissive_radius) * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_FLASH) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			float flash_radius = uLights[light_index].innerCutoff;
			float falloff_exp = uLights[light_index].outerCutoff;
			float norm_dist = distance / max(flash_radius, 0.001);
			attenuation = 1.0 / pow(1.0 + norm_dist, falloff_exp);
			float radius = 2.0 * flash_radius * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else {
			continue;
		}

		float NdotL = max(dot(n, L), 0.0);
		total_light += uLights[light_index].color * uLights[light_index].intensity * attenuation * NdotL;
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
		float attenuation = 1.0;

		if (uLights[light_index].type == LIGHT_TYPE_POINT) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_SPOT) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));

			float theta = dot(L, normalize(-uLights[light_index].direction));
			float epsilon = uLights[light_index].innerCutoff - uLights[light_index].outerCutoff;
			float angular_intensity = clamp((theta - uLights[light_index].outerCutoff) / epsilon, 0.0, 1.0);
			attenuation *= angular_intensity;
		} else if (uLights[light_index].type == LIGHT_TYPE_EMISSIVE) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			float emissive_radius = uLights[light_index].innerCutoff;
			float effective_dist = max(distance - emissive_radius * 0.5, 0.0);
			attenuation = 1.0 / (1.0 + 0.09 * effective_dist + 0.032 * effective_dist * effective_dist);
			float proximity_boost = smoothstep(emissive_radius * 2.0, 0.0, distance);
			attenuation = mix(attenuation, 1.0, proximity_boost * 0.5);
			float radius = (sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 + emissive_radius) * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_FLASH) {
			L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			float flash_radius = uLights[light_index].innerCutoff;
			float falloff_exp = uLights[light_index].outerCutoff;
			float norm_dist = distance / max(flash_radius, 0.001);
			attenuation = 1.0 / pow(1.0 + norm_dist, falloff_exp);
			float radius = 2.0 * flash_radius * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else {
			continue;
		}

		float NdotL = max(dot(n, L), 0.0);
		total_light += uLights[light_index].color * uLights[light_index].intensity * attenuation * NdotL;
	}

	return total_light;
}

/**
 * High-level GLSL helper to evaluate clustered local light contribution without normals.
 */
vec3 evaluateClusteredLightContributionSimple(vec3 frag_pos) {
	vec3 total_light = uAmbientLight.rgb;

	float exposureScale = 1.0;

	uint dir_count = min(uLightCount, 2u);
	for (uint i = 0u; i < dir_count; ++i) {
		if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL && uLights[i].intensity > 0.0) {
			total_light += uLights[i].color * uLights[i].intensity;
		}
	}

	Cluster global_cluster = uClusters[3456];
	for (uint i = 0; i < global_cluster.count; ++i) {
		uint light_index = global_cluster.lightIndices[i];
		if (uLights[light_index].intensity <= 0.0) continue;

		vec3 light_pos = uLights[light_index].position;
		if ((uLights[light_index].flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		float attenuation = 1.0;

		if (uLights[light_index].type == LIGHT_TYPE_POINT) {
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_SPOT) {
			vec3 L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));

			float theta = dot(L, normalize(-uLights[light_index].direction));
			float epsilon = uLights[light_index].innerCutoff - uLights[light_index].outerCutoff;
			float angular_intensity = clamp((theta - uLights[light_index].outerCutoff) / epsilon, 0.0, 1.0);
			attenuation *= angular_intensity;
		} else if (uLights[light_index].type == LIGHT_TYPE_EMISSIVE) {
			float distance = length(light_pos - frag_pos);
			float emissive_radius = uLights[light_index].innerCutoff;
			float effective_dist = max(distance - emissive_radius * 0.5, 0.0);
			attenuation = 1.0 / (1.0 + 0.09 * effective_dist + 0.032 * effective_dist * effective_dist);
			float proximity_boost = smoothstep(emissive_radius * 2.0, 0.0, distance);
			attenuation = mix(attenuation, 1.0, proximity_boost * 0.5);
			float radius = (sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 + emissive_radius) * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_FLASH) {
			float distance = length(light_pos - frag_pos);
			float flash_radius = uLights[light_index].innerCutoff;
			float falloff_exp = uLights[light_index].outerCutoff;
			float norm_dist = distance / max(flash_radius, 0.001);
			attenuation = 1.0 / pow(1.0 + norm_dist, falloff_exp);
			float radius = 2.0 * flash_radius * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else {
			continue;
		}

		total_light += uLights[light_index].color * uLights[light_index].intensity * attenuation;
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

		float attenuation = 1.0;

		if (uLights[light_index].type == LIGHT_TYPE_POINT) {
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_SPOT) {
			vec3 L = normalize(light_pos - frag_pos);
			float distance = length(light_pos - frag_pos);
			attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
			float radius = sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));

			float theta = dot(L, normalize(-uLights[light_index].direction));
			float epsilon = uLights[light_index].innerCutoff - uLights[light_index].outerCutoff;
			float angular_intensity = clamp((theta - uLights[light_index].outerCutoff) / epsilon, 0.0, 1.0);
			attenuation *= angular_intensity;
		} else if (uLights[light_index].type == LIGHT_TYPE_EMISSIVE) {
			float distance = length(light_pos - frag_pos);
			float emissive_radius = uLights[light_index].innerCutoff;
			float effective_dist = max(distance - emissive_radius * 0.5, 0.0);
			attenuation = 1.0 / (1.0 + 0.09 * effective_dist + 0.032 * effective_dist * effective_dist);
			float proximity_boost = smoothstep(emissive_radius * 2.0, 0.0, distance);
			attenuation = mix(attenuation, 1.0, proximity_boost * 0.5);
			float radius = (sqrt(max(0.1, uLights[light_index].intensity)) * 50.0 + emissive_radius) * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else if (uLights[light_index].type == LIGHT_TYPE_FLASH) {
			float distance = length(light_pos - frag_pos);
			float flash_radius = uLights[light_index].innerCutoff;
			float falloff_exp = uLights[light_index].outerCutoff;
			float norm_dist = distance / max(flash_radius, 0.001);
			attenuation = 1.0 / pow(1.0 + norm_dist, falloff_exp);
			float radius = 2.0 * flash_radius * exposureScale;
			attenuation *= smoothstep(1.0, 0.8, distance / max(radius, 0.001));
		} else {
			continue;
		}

		total_light += uLights[light_index].color * uLights[light_index].intensity * attenuation;
	}

	return total_light;
}

#endif // BRASSICA_CLUSTERED_LIGHTING_GLSL
