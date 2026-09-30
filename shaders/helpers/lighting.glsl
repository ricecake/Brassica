#ifndef HELPERS_LIGHTING_GLSL
#define HELPERS_LIGHTING_GLSL

#include "brdf.glsl"
#include "material.glsl"

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
		float planetRadius = FAKE_PLANET_RADIUS;
		vec3 planetCenter = vec3(0.0, -planetRadius, 0.0);
		vec3 fragToCenter = frag_pos - planetCenter;
		float distToCenter = length(fragToCenter);
		vec3 normalPlanet = fragToCenter / max(0.001, distToCenter);
		float NdotL = dot(normalPlanet, light_dir);
		attenuation = smoothstep(-0.05, 0.05, NdotL);
		if (NdotL <= -0.01) {
			attenuation = 0.0;
		} else {
			float t_closest = -dot(fragToCenter, light_dir);
			if (t_closest > 0.0) {
				float d2 = dot(fragToCenter, fragToCenter) - t_closest * t_closest;
				if (d2 < (planetRadius * planetRadius - 100.0)) {
					attenuation = 0.0;
				}
			}
		}
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
/**
 * Planet Horizon & Curvature Shadowing:
 * Evaluates whether a world-space point is shadowed by the curvature of the planet body relative to the sun direction.
 */
float calculatePlanetHorizonShadow(vec3 fragWorldPos, vec3 lightDir) {
	float planetRadius = FAKE_PLANET_RADIUS;
	vec3 planetCenter = vec3(0.0, -planetRadius, 0.0);
	vec3 fragToCenter = fragWorldPos - planetCenter;
	float distToCenter = length(fragToCenter);
	vec3 normalPlanet = fragToCenter / max(0.001, distToCenter);

	float NdotL = dot(normalPlanet, lightDir);
	if (NdotL <= -0.01) {
		return 0.0; // Beyond planet horizon
	}

	float t_closest = -dot(fragToCenter, lightDir);
	if (t_closest > 0.0) {
		float d2 = dot(fragToCenter, fragToCenter) - t_closest * t_closest;
		if (d2 < (planetRadius * planetRadius - 100.0)) {
			return 0.0; // Ray to sun passes through planet body
		}
	}

	return smoothstep(-0.01, 0.05, NdotL);
}

/**
 * Cascaded Shadow Map sampling and PCF evaluation.
 */
float evaluateCascadedShadow(vec3 fragWorldPos, float camDist) {
	float horizonFactor = calculatePlanetHorizonShadow(fragWorldPos, uSunDirection.xyz);
	if (horizonFactor <= 0.0) {
		return 0.0;
	}

	if (uShadowMapIndex == 0u) {
		return horizonFactor;
	}

	uint cascadeIdx = 3u;
	if (camDist < uCascadeSplits.x) cascadeIdx = 0u;
	else if (camDist < uCascadeSplits.y) cascadeIdx = 1u;
	else if (camDist < uCascadeSplits.z) cascadeIdx = 2u;
	else if (camDist > uCascadeSplits.w) return horizonFactor;

	vec4 lightPos = uCascadeViewProj[cascadeIdx] * vec4(fragWorldPos, 1.0);
	vec3 shadowCoords = lightPos.xyz / max(0.00001, lightPos.w);
	vec2 shadowUV = shadowCoords.xy * 0.5 + 0.5;

	if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0) {
		return horizonFactor;
	}

	float currentDepth = shadowCoords.z;
	if (currentDepth < 0.0 || currentDepth > 1.0) {
		return horizonFactor;
	}

	float shadow = 0.0;
	vec2 texelSize = vec2(1.0 / 1024.0);
	float bias = uShadowBias * (1.0 + float(cascadeIdx) * 0.5);

	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			vec2 sampleUV = shadowUV + vec2(x, y) * texelSize;
			vec4 texSample = SAMPLE_ARRAY_WRAP(uShadowMapIndex, vec3(sampleUV, float(cascadeIdx)));
			float sampledDepth = texSample.r;
			shadow += (currentDepth - bias <= sampledDepth) ? 1.0 : 0.0;
		}
	}
	shadow /= 9.0;

	return shadow * horizonFactor;
}

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
 * Accumulated output of the lighting system: total lit color, plus the specular-only luminance
 * broken out (previously computed and silently discarded) so a future glint/bloom hook has
 * something real to read instead of needing to re-derive it.
 */
struct LightingResult {
	vec3  color;
	float specularLuminance;
};

/**
 * Cook-Torrance BRDF evaluation for direct light. Takes the whole Material rather than unpacked
 * albedo/roughness/metallic so a new material property only needs to be read here, not added to
 * this signature and every call site -- see material.glsl.
 */
void evaluate_brdf(vec3 N, vec3 V, vec3 L, Material material, vec3 radiance, float shadow, inout LightingResult result) {
	float NdotL = max(dot(N, L), 0.0);
	if (NdotL <= 0.0) return;

	float NdotV = max(dot(N, V), 1e-4);
	vec3 H = normalize(V + L);
	float HdotV = max(dot(H, V), 0.0);
	vec3 F0 = mix(vec3(0.04), material.albedo, material.metallic);

	float NDF = DistributionGGX(N, H, material.roughness);
	float V_term = VisibilitySmithGGXCorrelated(NdotL, NdotV, material.roughness);
	vec3 F = fresnelSchlickFast(HdotV, F0);
	vec3 specular = NDF * V_term * F;

	vec3 kS = F;
	vec3 kD = (vec3(1.0) - kS) * (1.0 - material.metallic);

	vec3 specular_radiance = specular * radiance * NdotL * shadow;

	result.color += (kD * material.albedo / PI) * radiance * NdotL * shadow + specular_radiance;
	result.specularLuminance += get_luminance(specular_radiance);
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
