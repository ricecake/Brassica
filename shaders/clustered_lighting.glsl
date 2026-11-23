#ifndef BRASSICA_CLUSTERED_LIGHTING_GLSL
#define BRASSICA_CLUSTERED_LIGHTING_GLSL

#include "lighting.glsl"
#include "helpers/lighting.glsl"
#include "helpers/octahedral.glsl"
#include "atmosphere/common.glsl"
#include "helpers/whittaker.glsl"
#include "external/lygia/lighting/common/gtaoMultiBounce.glsl"

// getClusterIndex() now lives in lighting.glsl (this file's own #include above already pulls
// it in) -- shared with shaders/volumetric_froxel_gen.comp, which needs it without this file's
// heavier PBR/GTAO includes.

/**
 * High-level GLSL helper to evaluate clustered local light contribution with Cook-Torrance PBR BRDF.
 */
LightingResult evaluateClusteredLightContributionPBR(
	vec3  frag_pos,
	vec3  normal,
	Material material,
	uint  skyViewIndex,
	uint  weatherBiomeIndex,
	uint  gtaoIndex,
	vec2  uv,
	float aoDirectInfluence,
	uint  minMaxIndex,
	uvec4 gridParams, // x = numLODs, w = textureDim -- same layout as every terrain-aware pass's own push constants
	uint  cpuLightsIndex,
	uint  gpuLightsIndex
) {
	vec3 N = normalize(normal);
	vec3 V = normalize(uCameraPosition.xyz - frag_pos);

	LightingResult result = LightingResult(vec3(0.0), 0.0);

	// Evaluate directional lights (indices 0 and 1) -- always CPU-authored (sun/moon).
	uint dir_count = min(uCpuLightsBuffers[nonuniformEXT(cpuLightsIndex)].count, 2u);
	for (uint i = 0u; i < dir_count; ++i) {
		Light dirLight = uCpuLightsBuffers[nonuniformEXT(cpuLightsIndex)].lights[i];
		if (dirLight.type == LIGHT_TYPE_DIRECTIONAL && dirLight.intensity > 0.0) {
			vec3 L;
			float attenuation;
			calculateLightContribution(
				dirLight.type,
				dirLight.position,
				dirLight.direction,
				dirLight.innerCutoff,
				dirLight.outerCutoff,
				dirLight.intensity,
				frag_pos,
				L,
				attenuation
			);

			vec3 radiance = dirLight.color * (dirLight.intensity * PBR_INTENSITY_BOOST) * attenuation;
			evaluate_brdf(N, V, L, material, radiance, 1.0, result);
		}
	}

	// Evaluate global lights bucket (index 3456) -- CPU list first, then GPU list.
	Cluster global_cluster = uClusters[3456];
	for (uint i = 0; i < global_cluster.cpuCount; ++i) {
		uint light_index = global_cluster.cpuLightIndices[i];
		Light light = uCpuLightsBuffers[nonuniformEXT(cpuLightsIndex)].lights[light_index];
		if (light.intensity <= 0.0) continue;

		vec3 light_pos = light.position;
		if ((light.flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			light.type, light_pos, light.direction, light.innerCutoff, light.outerCutoff, light.intensity,
			frag_pos, L, attenuation
		);
		if (attenuation <= 0.0) continue;

		vec3 radiance = light.color * (light.intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}
	for (uint i = 0; i < global_cluster.count; ++i) {
		uint light_index = global_cluster.lightIndices[i];
		Light light = uGpuLightsBuffers[nonuniformEXT(gpuLightsIndex)].lights[light_index];
		if (light.intensity <= 0.0) continue;

		vec3 light_pos = light.position;
		if ((light.flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			light.type, light_pos, light.direction, light.innerCutoff, light.outerCutoff, light.intensity,
			frag_pos, L, attenuation
		);
		if (attenuation <= 0.0) continue;

		vec3 radiance = light.color * (light.intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}

	uint cluster_index = getClusterIndex(frag_pos);
	Cluster cluster = uClusters[cluster_index];

	for (uint i = 0; i < cluster.cpuCount; ++i) {
		uint light_index = cluster.cpuLightIndices[i];
		Light light = uCpuLightsBuffers[nonuniformEXT(cpuLightsIndex)].lights[light_index];
		if (light.intensity <= 0.0) continue;

		vec3 light_pos = light.position;
		if ((light.flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			light.type, light_pos, light.direction, light.innerCutoff, light.outerCutoff, light.intensity,
			frag_pos, L, attenuation
		);
		if (attenuation <= 0.0) continue;

		vec3 radiance = light.color * (light.intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}
	for (uint i = 0; i < cluster.count; ++i) {
		uint light_index = cluster.lightIndices[i];
		Light light = uGpuLightsBuffers[nonuniformEXT(gpuLightsIndex)].lights[light_index];
		if (light.intensity <= 0.0) continue;

		vec3 light_pos = light.position;
		if ((light.flags & LIGHT_FLAG_CAMERA_RELATIVE) != 0) {
			light_pos += uCameraPosition.xyz;
		}

		vec3 L;
		float attenuation;
		calculateLightContribution(
			light.type, light_pos, light.direction, light.innerCutoff, light.outerCutoff, light.intensity,
			frag_pos, L, attenuation
		);
		if (attenuation <= 0.0) continue;

		vec3 radiance = light.color * (light.intensity * PBR_INTENSITY_BOOST) * attenuation;
		evaluate_brdf(N, V, L, material, radiance, 1.0, result);
	}

	// terrainClipmapLevelFor can return numLevels itself as an "out of clipmap coverage" sentinel
	// (its own comment in terrain.glsl), not a valid array-layer index -- every other caller
	// (traceTerrainOcclusion, calculateTerrainSoftShadow) clamps to numLevels-1 before using it;
	// this one didn't, so most distant terrain (anything past LOD 0's own half-extent) fed an
	// out-of-bounds layer index into sampleTerrainMinMaxMip, reading undefined data and collapsing
	// terrainOcc toward 0 almost everywhere.
	uint  numLevels = max(gridParams.x, 1u);
	uint  terrainOccLevel = min(terrainClipmapLevelFor(frag_pos.xz, gridParams.w, gridParams.x), numLevels - 1u);
	float terrainOcc = calculateTerrainOcclusion(frag_pos, minMaxIndex, terrainOccLevel, gridParams.w);

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

	// AO is physically an indirect-light-only effect, but indirect light is a small fraction
	// of outdoor daylight here, so a pure indirect-only application is nearly invisible.
	// aoDirectInfluence lets a configurable fraction of it read as contact shadowing on
	// direct-lit surfaces too, purely for visual legibility of contouring/enclosure.
	result.color *= mix(vec3(1.0), multiBounceAO, clamp(aoDirectInfluence, 0.0, 1.0));

	vec3 spatialSHAmbient = getSpatialAmbientSH(frag_pos, N);
	vec3 finalAmbient = (spatialSHAmbient + ambientLight) * uAmbientLight.rgb;

	result.color += finalAmbient * material.albedo * multiBounceAO * terrainOcc;
	result.color += material.albedo * material.emissivity;

	return result;
}

LightingResult evaluateClusteredLightContributionPBR(vec3 frag_pos, vec3 normal, Material material) {
	return evaluateClusteredLightContributionPBR(
		frag_pos, normal, material, 0u, 0u, 0u, vec2(0.5), 0.0, 0u, uvec4(0), 0u, 0u
	);
}

#endif // BRASSICA_CLUSTERED_LIGHTING_GLSL
