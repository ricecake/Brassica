#ifndef HELPERS_LIGHTING_GLSL
#define HELPERS_LIGHTING_GLSL

#include "brdf.glsl"

#ifndef LIGHTING_TYPES
#define LIGHTING_TYPES
const int LIGHT_TYPE_POINT = 0;
const int LIGHT_TYPE_DIRECTIONAL = 1;
const int LIGHT_TYPE_SPOT = 2;
const int LIGHT_TYPE_EMISSIVE = 3; // Glowing object light
const int LIGHT_TYPE_FLASH = 4;    // Explosion/flash light

// Light flags
const int LIGHT_FLAG_CASTS_SHADOW = 1;
const int LIGHT_FLAG_VOLUMETRIC_SHADOW = 2;
const int LIGHT_FLAG_CAMERA_RELATIVE = 4;
const int LIGHT_FLAG_CLOUD_EMISSIVE = 8;
#endif
// PBR intensity multiplier
const float PBR_INTENSITY_BOOST = 1.0;

/**
 * Calculate light direction and attenuation for any light type.
 */
void calculateLightContribution(
	int type,
	vec3 light_pos,
	vec3 light_dir_param,
	float inner_cutoff,
	float outer_cutoff,
	float intensity,
	vec3 frag_pos,
	out vec3 light_dir,
	out float attenuation
) {
	attenuation = 1.0;

	if (type == LIGHT_TYPE_POINT) {
		light_dir = normalize(light_pos - frag_pos);
		float distance = length(light_pos - frag_pos);
		attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
		if (outer_cutoff > 0.0) {
			attenuation *= smoothstep(1.0, 0.8, distance / max(outer_cutoff, 0.001));
		}
	} else if (type == LIGHT_TYPE_DIRECTIONAL) {
		light_dir = normalize(-light_dir_param);
		attenuation = 1.0;
	} else if (type == LIGHT_TYPE_SPOT) {
		light_dir = normalize(light_pos - frag_pos);
		float distance = length(light_pos - frag_pos);
		attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
		if (outer_cutoff > 0.0) {
			attenuation *= smoothstep(1.0, 0.8, distance / max(outer_cutoff, 0.001));
		}

		float theta = dot(light_dir, normalize(-light_dir_param));
		float epsilon = inner_cutoff - outer_cutoff;
		float angular_intensity = clamp((theta - outer_cutoff) / max(epsilon, 0.0001), 0.0, 1.0);
		attenuation *= angular_intensity;
	} else if (type == LIGHT_TYPE_EMISSIVE) {
		light_dir = normalize(light_pos - frag_pos);
		float distance = length(light_pos - frag_pos);
		float emissive_radius = inner_cutoff;
		float effective_dist = max(distance - emissive_radius * 0.5, 0.0);
		attenuation = 1.0 / (1.0 + 0.09 * effective_dist + 0.032 * effective_dist * effective_dist);
		float proximity_boost = smoothstep(emissive_radius * 2.0, 0.0, distance);
		attenuation = mix(attenuation, 1.0, proximity_boost * 0.5);
	} else if (type == LIGHT_TYPE_FLASH) {
		light_dir = normalize(light_pos - frag_pos);
		float distance = length(light_pos - frag_pos);
		float flash_radius = inner_cutoff;
		float falloff_exp = outer_cutoff > 0.0 ? outer_cutoff : 2.0;
		float norm_dist = distance / max(flash_radius, 0.001);
		attenuation = 1.0 / pow(1.0 + norm_dist, falloff_exp);
		attenuation *= smoothstep(2.0, 1.5, norm_dist);
	} else {
		light_dir = vec3(0.0, 1.0, 0.0);
		attenuation = 0.0;
	}
}

/**
 * Spatial Ambient SH probes integration point.
 * Evaluates global SH irradiance or per-chunk ambient probes.
 */
#ifndef SPATIAL_AMBIENT_SH_DEFINED
#define SPATIAL_AMBIENT_SH_DEFINED
vec3 getSpatialAmbientSH(vec3 worldPos, vec3 N) {
	// Fallback to evaluating ambient diffuse from normal
	return vec3(0.15); // Will be driven by LightingUBO ambient / SH coefficients
}
#endif

/**
 * Macro terrain occlusion integration point.
 */
#ifndef TERRAIN_OCCLUSION_DEFINED
#define TERRAIN_OCCLUSION_DEFINED
float calculateTerrainOcclusion(vec3 worldPos, vec3 normal) {
	return 1.0;
}
#endif

/**
 * Relative luminance calculation.
 */
float get_luminance(vec3 color) {
	return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/**
 * Cook-Torrance BRDF evaluation for direct light.
 */
void evaluate_brdf(
	vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic, vec3 F0,
	vec3 radiance, float shadow, inout vec3 Lo, inout float spec_lum)
{
	float NdotL = max(dot(N, L), 0.0);
	if (NdotL <= 0.0) return;

	float NdotV = max(dot(N, V), 1e-4);
	vec3 H = normalize(V + L);
	float HdotV = max(dot(H, V), 0.0);

	float NDF = DistributionGGX(N, H, roughness);
	float V_term = VisibilitySmithGGXCorrelated(NdotL, NdotV, roughness);
	vec3 F = fresnelSchlickFast(HdotV, F0);
	vec3 specular = NDF * V_term * F;

	vec3 kS = F;
	vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

	vec3 specular_radiance = specular * radiance * NdotL * shadow;

	Lo += (kD * albedo / PI) * radiance * NdotL * shadow + specular_radiance;
	spec_lum += get_luminance(specular_radiance);
}

/**
 * Foliage BRDF evaluation (Subsurface transmission approximation).
 */
void evaluate_foliage_brdf(
	vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic, vec3 F0,
	vec3 radiance, float shadow, float translucency, inout vec3 Lo, inout float spec_lum)
{
	float NdotL = dot(N, L);
	float NdotV = max(dot(N, V), 1e-4);

	if (NdotL > 0.0) {
		vec3 H = normalize(V + L);
		float HdotV = max(dot(H, V), 0.0);

		float NDF = DistributionGGX(N, H, roughness);
		float V_term = VisibilitySmithGGXCorrelated(NdotL, NdotV, roughness);
		vec3 F = fresnelSchlickFast(HdotV, F0);

		vec3 specular = NDF * V_term * F;
		vec3 kS = F;
		vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

		vec3 specular_out = specular * radiance * NdotL * shadow;
		Lo += (kD * albedo / PI) * radiance * NdotL * shadow + specular_out;
		spec_lum += get_luminance(specular_out);
	}

	float transmissionNdotL = max(-NdotL, 0.0);
	if (transmissionNdotL > 0.0) {
		vec3 transmission_out = (albedo / PI) * radiance * transmissionNdotL * translucency * shadow;
		Lo += transmission_out;
	}
}

/**
 * Calculate iridescence color based on view angle.
 */
vec3 calculate_iridescence(vec3 view_dir, vec3 normal, vec3 base_color, float time_offset) {
	float NdotV = abs(dot(view_dir, normal));
	float angle_factor = pow(1.0 - NdotV, 2.0);

	float swirl = sin(time_offset * 0.5) * 0.5 + 0.5;
	vec3 iridescent = vec3(
		sin(angle_factor * 10.0 + swirl * 5.0) * 0.5 + 0.5,
		sin(angle_factor * 10.0 + swirl * 5.0 + 2.0) * 0.5 + 0.5,
		sin(angle_factor * 10.0 + swirl * 5.0 + 4.0) * 0.5 + 0.5
	);

	return mix(base_color, iridescent, angle_factor * 0.8 + 0.2);
}

/**
 * Emissive glow calculation.
 */
vec3 calculate_emission(vec3 base_emission, float intensity, float falloff) {
	return base_emission * intensity * (1.0 + falloff);
}

/**
 * Flash/explosion illumination contribution.
 */
vec3 calculate_flash_contribution(
	vec3 frag_pos,
	vec3 normal,
	vec3 flash_pos,
	vec3 flash_color,
	float flash_intensity,
	float flash_radius,
	float flash_time
) {
	vec3 L = normalize(flash_pos - frag_pos);
	float distance = length(flash_pos - frag_pos);
	float NdotL = max(dot(normalize(normal), L), 0.0);

	float norm_dist = distance / max(flash_radius, 0.001);
	float dist_atten = 1.0 / pow(1.0 + norm_dist, 2.0);
	dist_atten *= smoothstep(2.0, 1.0, norm_dist);

	float time_atten = pow(1.0 - clamp(flash_time, 0.0, 1.0), 3.0);

	return flash_color * flash_intensity * dist_atten * time_atten * NdotL;
}

#endif // HELPERS_LIGHTING_GLSL
