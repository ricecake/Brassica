#version 460
#include "bindless.glsl"
#include "common.glsl"
#include "clustered_lighting.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform WaterPushConstants {
	uvec4 gridParams;
	vec3  waterColor;
	float waterLevel;
	uint  gPositionIndex;
	uint  gAlbedoIndex;
	uint  gNormalIndex;
	uint  sceneColorIndex;
	uint  baseLOD;
	uint  maxLODs;
} params;

void main() {
	// 1. DETERMINE VIEW ORIENTATION
	bool isAboveWater = gl_FrontFacing;

	ivec2 gTexSize = textureSize(
		sampler2D(uTextures2D[nonuniformEXT(params.gAlbedoIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]),
		0
	);
	vec2 screenUV = gl_FragCoord.xy / vec2(gTexSize);

	vec4 albedo = SAMPLE_NEAREST(params.gAlbedoIndex, screenUV);
	vec4 sceneColor = SAMPLE_NEAREST(params.sceneColorIndex, screenUV);
	vec3 relTerrainPos = SAMPLE_NEAREST(params.gPositionIndex, screenUV).rgb;
	vec3 relWaterPos = inWorldPos;
	vec3 absWaterPos = uCameraPosition.xyz + inWorldPos;

	float distToCamWater = length(relWaterPos);
	float rayLengthThroughWater = 100.0;
	float depthBelowWater = 100.0;

	float distToCamTerrain = length(relTerrainPos);
	if (albedo.a >= 0.01) {
		if (distToCamTerrain < distToCamWater - 0.2) {
			discard;
		}
	}


	// 2. DUAL-SIDED OPTICAL DEPTH CALCULATION
	if (albedo.a >= 0.01) {

		if (isAboveWater) {
			// Looking DOWN into the water
			if (distToCamTerrain <= distToCamWater) {
				discard; // Terrain is strictly in front of the water mesh
			}

			rayLengthThroughWater = max(0.0, distToCamTerrain - distToCamWater);
			depthBelowWater = relWaterPos.y - relTerrainPos.y;

			if (depthBelowWater <= 0.0 && rayLengthThroughWater <= 0.0) {
				discard;
			}
		} else {
			// Looking UP at the water surface from below
			// The view ray travels entirely through the water volume from camera to surface
			rayLengthThroughWater = distToCamWater;
			depthBelowWater = params.waterLevel - uCameraPosition.y;
		}
	} else {
		// Sky background
		rayLengthThroughWater = isAboveWater ? 100.0 : distToCamWater;
		depthBelowWater = isAboveWater ? 100.0 : (params.waterLevel - uCameraPosition.y);
	}

	vec3 baseNormal = normalize(inNormal);
	if (!isAboveWater) {
		baseNormal = -baseNormal; // Flip normal if viewing from below
	}
	if (length(baseNormal) < 0.1) {
		baseNormal = isAboveWater ? vec3(0.0, 1.0, 0.0) : vec3(0.0, -1.0, 0.0);
	}

	// vec2 waveXZ = inWorldPos.xz;
	// float t = uTime * smoothstep(800.0, 1800.0, distToCamWater) * max(1.0, 1000.0/distToCamWater);

	// vec2 sinGrad = 0.25 * cross_noise_fbm(inWorldPos * 0.004 + abs(dot_noise(inWorldPos * 0.0025, t * 0.5)), 4, t * 0.25).xz;
	// sinGrad *= smoothstep(1000.0, 2000.0, distToCamWater) * (1.0 - smoothstep(3000.0, 20000.0, distToCamWater));

	// Ensure wave perturbation follows the flipped backface normal
	vec3 waveNormal = baseNormal;//normalize(baseNormal + vec3(-sinGrad.x, 0.0, -sinGrad.y));

	float distToCam = distToCamWater;
	float closeThreshold = 800.0;
	float closeFactor = clamp(1.0 - distToCam / closeThreshold, 0.0, 1.0);

	// 3. PHYSICAL REFRACTION & TOTAL INTERNAL REFLECTION
	vec3 viewDir = normalize(-relWaterPos);
	float iorRatio = isAboveWater ? (1.0 / 1.333) : (1.333 / 1.0);
	vec3 refractDir = refract(-viewDir, waveNormal, iorRatio);

	vec2 refractOffset = vec2(0.0);
	bool isTIR = false;

	if (isAboveWater) {
		refractOffset = waveNormal.xz * clamp(rayLengthThroughWater * 0.008, 0.0, 0.03);
	} else {
		if (length(refractDir) < 0.01) {
			isTIR = true; // Total Internal Reflection
		} else {
			refractOffset = refractDir.xz * 0.05; // Refracting the sky above
		}
	}

	vec2 refractUV = clamp(screenUV + refractOffset, vec2(0.0), vec2(1.0));

	vec4 refractedAlbedo = SAMPLE_NEAREST(params.sceneColorIndex, refractUV);
	vec3 relRefractedPos = SAMPLE_NEAREST(params.gPositionIndex, refractUV).rgb;

	if (isAboveWater && (relWaterPos.y - relRefractedPos.y <= 0.0 || refractedAlbedo.a < 0.01)) {
		refractedAlbedo = sceneColor;
		relRefractedPos = relTerrainPos; // Ensure position falls back too
	}

	if (isTIR && refractedAlbedo.a >= 0.01) {
		float surfaceToTerrain = length(relRefractedPos - relWaterPos);
		rayLengthThroughWater = distToCamWater + surfaceToTerrain;
	}

	vec3 extinctionCoeff = vec3(0.28, 0.07, 0.02);
	vec3 transmittance = exp(-extinctionCoeff * rayLengthThroughWater);

	vec3 shallowWaterTint = vec3(0.12, 0.62, 0.78);
	vec3 deepWaterColor = vec3(0.01, 0.12, 0.32);
	vec3 baseTint = mix(shallowWaterTint, params.waterColor, clamp(depthBelowWater / 5.0, 0.0, 1.0));

	// Out-scattering color of the water volume itself
	vec3 waterBodyColor = mix(deepWaterColor, baseTint, transmittance);

	// 4. MANUAL COMPOSITE (Bypassing fixed-function blend)
	// Absorb the background light and add the water's scattered light
	vec3 integratedColor = (refractedAlbedo.rgb * transmittance) + (waterBodyColor * (1.0 - transmittance));

	// Water has no Lambertian diffuse term of its own (its visible color comes entirely from the
	// volume-scattering/transmittance math above) -- zero albedo means evaluate_brdf's diffuse
	// lobe contributes nothing and this reduces to a real Cook-Torrance specular highlight driven
	// by every light in range (sun, moon, points/spots via the cluster grid), not just a hardcoded
	// single directional "sun" reimplementing Blinn-Phong. Low roughness keeps the highlight tight,
	// matching the old pow(NdotH, 256) sharpness; the wave normal's own noise perturbation already
	// carries the surface's visual roughness.
	Material waterMaterial = Material(vec3(0.0), 0.05, 0.0, 1.0);
	vec3     shineColor =
		isAboveWater ? evaluateClusteredLightContributionPBR(absWaterPos, waveNormal, waterMaterial).color : vec3(0.0);

	float NdotV = max(dot(viewDir, waveNormal), 0.0);
	float fresnel = clamp(pow(1.0 - NdotV, 5.0), 0.02, 0.98);

	if (isTIR) {
		fresnel = 1.0;
	}

	vec3 surfaceReflectionColor = isAboveWater ? vec3(0.65, 0.82, 1.0) : deepWaterColor;
	vec3 finalColor = mix(integratedColor, surfaceReflectionColor, fresnel * 0.5) + shineColor;

	float shoreFoam = clamp(1.0 - depthBelowWater / 2.2, 0.0, 1.0);
	shoreFoam = pow(shoreFoam, 1.4);
	float foamNoise = InterleavedGradientNoise(absWaterPos.xz * 3.5, int(uTime * 12.0));
	shoreFoam *= 0.65 + 0.35 * foamNoise;

	float crestFactor = clamp((1.0 - waveNormal.y) * 3.5, 0.0, 1.0);
	float crestFoam = crestFactor * closeFactor;// * (0.5 + 0.5 * sin(uTime * 3.0 + inWorldPos.x * 0.5));

	float totalFoam = clamp(shoreFoam * 1.25 + crestFoam * 0.6, 0.0, 1.0);
	if (!isAboveWater) totalFoam *= 0.15; // Diminish foam visibility heavily from underneath
	// totalFoam = 0.0;
	vec3 foamColor = vec3(0.92, 0.96, 1.0);
	finalColor = mix(finalColor, foamColor, totalFoam);

	// Output completely opaque fragment to overwrite the G-Buffer composite
	outColor = vec4(finalColor, 1.0);
}