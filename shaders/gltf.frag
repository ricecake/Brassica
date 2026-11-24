#version 460

#include "helpers/material_system.glsl"
#include "common.glsl"

layout(location = 0) in vec3      vWorldPos;
layout(location = 1) in vec3      vNormal;
layout(location = 2) in vec2      vUv;
layout(location = 5) in vec3 vRestPos;
layout(location = 6) in vec3 vRestNormal;
layout(location = 3) flat in uint vMaterialIndex;
layout(location = 4) flat in uint vMaterialBufferIndex;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

const float WIND_STRENGTH = 0.85; // 0.0 = calm, 1.0 = steady breeze

// Precision-safe 3D hash (Dave Hoskins)
vec3 hash33(vec3 p3) {
	p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yxz + 33.33);
	return fract((p3.xxy + p3.yxx) * p3.zyx);
}

struct LeafSample {
	vec3  normal;
	vec3  cellId;
	float edgeMask;   // 0 at leaf border/crevice, 1 in leaf interior
	float midrib;     // 1 along center vein, 0 elsewhere
	float radialDist; // 0 at leaf center, ~1 at tip/edge
};

// Cheap 3D value noise (trilinear-interpolated hash lattice) -- used only to domain-warp the
// leaf cell search below, distinct from hash33's raw per-cell randomness.
vec3 valueNoise3(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	vec3 u = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(mix(hash33(i + vec3(0.0, 0.0, 0.0)), hash33(i + vec3(1.0, 0.0, 0.0)), u.x),
			mix(hash33(i + vec3(0.0, 1.0, 0.0)), hash33(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
		mix(mix(hash33(i + vec3(0.0, 0.0, 1.0)), hash33(i + vec3(1.0, 0.0, 1.0)), u.x),
			mix(hash33(i + vec3(0.0, 1.0, 1.0)), hash33(i + vec3(1.0, 1.0, 1.0)), u.x), u.y),
		u.z);
}

// Generates overlapping, randomly oriented 3D leaf blades across a solid hull. warp domain-warps
// the cell search (0 = none, matching the old hardcoded behavior); sharpness controls the
// edgeMask's cell/edge crispness (0.14 was the old hardcoded smoothstep width).
LeafSample evaluateLeafCanopy(vec3 p, vec3 macroNormal, float warp, float sharpness) {
	p += (valueNoise3(p * 0.37) - 0.5) * warp;

	vec3 n = floor(p);
	vec3 f = fract(p);

	float f1 = 8.0;
	float f2 = 8.0;
	vec3  bestId     = vec3(0.0);
	vec3  bestNormal = macroNormal;
	vec3  bestLocalR = vec3(0.0);

	for (int k = -1; k <= 1; k++) {
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				vec3 g = vec3(float(i), float(j), float(k));
				vec3 cellCoord = n + g;
				vec3 o = hash33(cellCoord);

				// Randomize individual leaf normal, biased outward along the hull normal
				vec3 randDir = normalize(hash33(cellCoord + 19.19) - 0.5);
				vec3 leafN   = normalize(macroNormal * 0.85 + randDir * 0.75);

				// Build an orthonormal tangent frame for this leaf and spin it randomly
				vec3 ref   = abs(leafN.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
				vec3 baseT = normalize(cross(ref, leafN));
				vec3 baseB = cross(leafN, baseT);

				float angle = o.z * 6.2831853;
				float ca = cos(angle), sa = sin(angle);
				vec3 T = baseT * ca + baseB * sa;
				vec3 B = -baseT * sa + baseB * ca;

				// Anisotropic projection turns isotropic Voronoi blobs into elongated leaf blades
				vec3 r = g + o - f;
				vec3 localR = vec3(
					dot(r, T) * 0.55,     // Elongate along leaf length
					dot(r, B) * 1.15,     // Narrow across leaf width
					dot(r, leafN) * 1.45  // Flatten leaf thickness
				);

				float d = length(localR);
				if (d < f1) {
					f2 = f1;
					f1 = d;
					bestId     = o;
					bestNormal = leafN;
					bestLocalR = localR;
				} else if (d < f2) {
					f2 = d;
				}
			}
		}
	}

	LeafSample res;
	res.normal     = bestNormal;
	res.cellId     = bestId;
	// max() guards smoothstep's edge0 < edge1 requirement -- sharpness could be driven to 0 by a
	// material author.
	res.edgeMask   = smoothstep(0.0, max(sharpness, 1e-4), f2 - f1);
	res.midrib     = 1.0 - smoothstep(0.0, 0.06, abs(bestLocalR.y));
	res.radialDist = clamp(f1, 0.0, 1.0);
	return res;
}

struct BarkSample {
	vec3  normal;
	float ridgeMask; // 0 in a groove/fissure, 1 on a ridge crest
};

// Generates vertical bark ridges/fissures running along world-space Y -- an approximation of
// the trunk axis (this engine's procedural trees are built trunk-up, see
// ProceduralArchetypeCatalog/ProceduralPlantGenerator), not the mesh's actual local axis, which
// isn't available here. warp domain-warps the ridge pattern so it doesn't look perfectly
// straight; sharpness controls the crest/groove transition width.
BarkSample evaluateBarkRidges(vec3 p, vec3 macroNormal, float warp, float sharpness) {
	// Stretch Y so ridges run long and vertical; keep X/Z tight so grooves tile around the trunk.
	vec3 q = vec3(p.x, p.y * 0.01, p.z);
	q += (valueNoise3(q * 0.6) - 0.5) * warp;
	q*=15.0;

	float ridge = valueNoise3(q).x;
	float crest = smoothstep(0.5 - max(sharpness, 1e-4), 0.5 + max(sharpness, 1e-4), ridge);

	// Groove normal perturbation via a central-difference gradient of the same ridge field,
	// carved into the surface along the circumferential (non-Y) tangent directions.
	float ridgeDx = valueNoise3(q + vec3(0.05, 0.0, 0.0)).x - ridge;
	float ridgeDz = valueNoise3(q + vec3(0.0, 0.0, 0.05)).x - ridge;
	vec3  tangent   = normalize(cross(macroNormal, vec3(0.0, 1.0, 0.0)) + vec3(1e-4));
	vec3  bitangent = cross(tangent, macroNormal);
	vec3  bumped    = macroNormal - tangent * ridgeDx * 4.0 - bitangent * ridgeDz * 4.0;

	BarkSample res;
	res.normal    = normalize(bumped);
	res.ridgeMask = crest;
	return res;
}

struct RockSample {
	vec3  normal;
	float edgeMask; // 0 at the seam between facets, 1 in a facet's flat interior
};

// Generates a faceted, low-poly-looking rock surface -- flat per-cell normals from an isotropic
// 3D Worley search, unlike evaluateLeafCanopy's anisotropic blade projection, so facets read as
// flat tilted planes rather than blobs. warp domain-warps the facet boundaries; sharpness
// controls the seam crispness.
RockSample evaluateRockFacets(vec3 p, vec3 macroNormal, float warp, float sharpness) {
	p += (valueNoise3(p * 0.4) - 0.5) * warp;

	vec3 n = floor(p);
	vec3 f = fract(p);

	float f1 = 8.0;
	float f2 = 8.0;
	vec3  bestNormal = macroNormal;

	for (int k = -1; k <= 1; k++) {
		for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				vec3 g = vec3(float(i), float(j), float(k));
				vec3 cellCoord = n + g;
				vec3 o = hash33(cellCoord);

				// Each facet gets one flat, randomly tilted normal, biased toward the macro
				// normal so facets still read as the same rough shape rather than noise soup.
				vec3 randDir = normalize(hash33(cellCoord + 7.77) - 0.5);
				vec3 facetN  = normalize(macroNormal * 1.3 + randDir * 0.6);

				float d = length(g + o - f);
				if (d < f1) {
					f2 = f1;
					f1 = d;
					bestNormal = facetN;
				} else if (d < f2) {
					f2 = d;
				}
			}
		}
	}

	RockSample res;
	res.normal   = bestNormal;
	res.edgeMask = smoothstep(0.0, max(sharpness, 1e-4), f2 - f1);
	return res;
}
/*
vec3 shadeLeaf() {
	// 4. Per-leaf rotational shimmer (normal perturbation)
	float leafPhase = leaf.cellId.x * 23.7 + leaf.cellId.y * 41.3;
	float leafFreq  = mix(4.5, 8.5, leaf.cellId.z);
	float tiltAngle = sin(iTime * leafFreq + leafPhase) * 0.28 * gust;

	vec3 windTangent = normalize(cross(windDir, leaf.normal) + vec3(1e-4));
	vec3 N = normalize(leaf.normal + windTangent * tiltAngle);

	// 5. Procedural leaf albedo (per-leaf hue shift + vein + canopy depth)
	vec3 deepGreen   = vec3(0.07, 0.22, 0.06);
	vec3 freshGreen  = vec3(0.18, 0.42, 0.11);
	vec3 sunTipGreen = vec3(0.34, 0.52, 0.14);

	vec3 albedo = mix(deepGreen, freshGreen, leaf.cellId.x);
	albedo = mix(albedo, sunTipGreen, leaf.cellId.y * smoothstep(-0.2, 0.8, N_macro.y));
	albedo = mix(albedo, sunTipGreen * 1.15, leaf.midrib * 0.35);

	// 6. Lighting: wrapped diffuse + backlit translucency + waxy leaf specular + crevice AO
	float NdotL     = dot(N, L);
	float wrapDiff  = max(0.0, (NdotL + 0.45) / 1.45);
	float backLight = pow(clamp(dot(V, -L + N * 0.35), 0.0, 1.0), 3.0) * (0.4 + 0.6 * leaf.radialDist);

	// Dappled overhead canopy shadow modulation
	float canopyDapple = 0.85 + 0.15 * sin(dot(worldPos.xz, windDir.xz) * 2.5 - iTime * 3.0);

	// Waxy cuticle specular highlight
	vec3  H    = normalize(L + V);
	float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.22 * leaf.edgeMask;

	// Ambient occlusion in crevices between overlapping leaf cells and underside of hull
	float ao = mix(0.45, 1.0, leaf.edgeMask) * mix(0.65, 1.0, N_macro.y * 0.5 + 0.5);

	vec3 directLight  = sunColor * (wrapDiff + backLight * 0.55) * canopyDapple;
	vec3 ambientLight = skyColor * 0.35 * ao;
	return albedo * (directLight + ambientLight) + sunColor * spec * canopyDapple;
}
*/

vec3 rotateNormalRestToWorld(vec3 nPerturbedRest, vec3 nRest, vec3 nWorld) {
	float c = dot(nRest, nWorld);
	if (c < -0.9999) {
		vec3 ortho = abs(nRest.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
		vec3 u = normalize(cross(nRest, ortho));
		return 2.0 * dot(u, nPerturbedRest) * u - nPerturbedRest;
	}
	vec3 axis = cross(nRest, nWorld);
	return normalize(
		nPerturbedRest +
		cross(axis, nPerturbedRest) +
		cross(axis, cross(axis, nPerturbedRest)) / (1.0 + c)
	);
}

void main() {
	MaterialData mat = uMaterialBuffers[nonuniformEXT(vMaterialBufferIndex)].data[vMaterialIndex];

	vec4  baseColor = evaluateGltfBaseColor(mat, vUv);
	vec2  metallicRoughness = evaluateGltfMetallicRoughness(mat, vUv);
	float occlusion = evaluateGltfOcclusion(mat, vUv);

	// vWorldPos is NOT camera-relative, despite this comment's own prior claim -- it's absolute
	// world position. model_render.mesh's outWorldPos (and the old, now-deleted gltf_render.mesh
	// before it) computes a genuinely camera-relative relWorldPos for gl_Position, then adds
	// uCameraPosition.xyz *back* for this varying specifically, same reasoning foliage.mesh's own
	// per-vertex outWorldPos writes do -- but ball.frag/foliage.frag both subtract that camera
	// position back out before writing outPosition, while this shader never did, so every model
	// (procedural decor and literal glTF alike) has been writing absolute position into
	// GBufferPosition. Every GBufferPosition consumer that treats it as relPos and does
	// length(relPos) (the whole cloud_raymarch.comp/cloud_temporal.comp/cloud_composite.frag
	// chain, matching gtao.comp's own explicit "camera-relative" contract comment) was computing
	// distance-from-world-origin instead of distance-from-camera for every model pixel -- on a
	// planet-scale world (R=600000) almost always far larger than any real cloud depth, which
	// defeats cloud occlusion against model-rendered geometry uniformly, across every archetype,
	// regardless of wind/culling/VRS (sgreenhusted confirmed none of those explain it alone).
	vec3 relPos = vWorldPos - uCameraPosition.xyz;
	outPosition = vec4(relPos, 0.0); // w = emissivity; 0.0 matches every non-foliage GBuffer
									  // writer (see foliage.frag's own emissivity fix this session).
									  // Real additive glTF emissive isn't composited here yet --
									  // see material_system.glsl's own comment on why.
	// evaluateLeafCanopy used to run unconditionally for every fragment through this shader --
	// bark, rock, flowers, and real glTF models included -- clobbering their normal and AO.
	// Gated on material_role so only materials explicitly registered for a procedural role get it.
	vec3 nWorld = normalize(vNormal);
	vec3 nRest  = normalize(vRestNormal);
	vec3 shadingNormal = nWorld;
	float ao = 1.0;

	if (mat.material_role == MATERIAL_ROLE_PROCEDURAL_LEAF) {
		// Wrap vWorldPos to preserve fp32 precision at planet-scale coordinates
		vec3 wPosWrapped = mod(vWorldPos, 64.0);

		// 1. Subtle world-space domain displacement keeps leaves bound to vRestPos
		//    while shifting blade overlaps as the branch moves through space
		vec3 worldWiggle = (valueNoise3(wPosWrapped * 2.5) - 0.5) * 0.5;
		vec3 leafSamplePos = vRestPos * mat.noise_scale + worldWiggle;

		LeafSample ls = evaluateLeafCanopy(leafSamplePos, nRest, mat.noise_warp, mat.noise_sharpness);

		// 2. Per-leaf normal flutter driven by spatial traversal through vWorldPos
		float leafPhase  = ls.cellId.x * 23.7 + ls.cellId.y * 41.3;
		float worldPhase = dot(wPosWrapped, vec3(3.1, 4.7, 2.9)) + leafPhase;
		float tiltAngle  = sin(worldPhase) * 0.25 * WIND_STRENGTH;

		vec3 flutterAxis = normalize(cross(nRest, vec3(0.0, 1.0, 0.0)) + vec3(1e-4));
		vec3 tiltedRestN = normalize(ls.normal + flutterAxis * tiltAngle);
		vec3 leafWorldN  = rotateNormalRestToWorld(tiltedRestN, nRest, nWorld);

		ao = mix(0.45, 1.0, ls.edgeMask) * mix(0.65, 1.0, nWorld.y * 0.5 + 0.5);
		shadingNormal = normalize(mix(nWorld, leafWorldN, mat.procedural_shading.x));

		vec3 leafColor = evaluateCosinePalette(mat, ls.cellId.x) * mix(1.0, 0.65, ls.midrib);
		baseColor.rgb = leafColor;

	} else if (mat.material_role == MATERIAL_ROLE_PROCEDURAL_BARK) {
		BarkSample bs = evaluateBarkRidges(vRestPos * mat.noise_scale, nRest, mat.noise_warp, mat.noise_sharpness);
		vec3 barkWorldN = rotateNormalRestToWorld(bs.normal, nRest, nWorld);

		ao = mix(0.5, 1.0, bs.ridgeMask);
		shadingNormal = normalize(mix(nWorld, barkWorldN, mat.procedural_shading.x));
		baseColor.rgb = evaluateCosinePalette(mat, 1.0 - bs.ridgeMask);

	} else if (mat.material_role == MATERIAL_ROLE_PROCEDURAL_ROCK) {
		RockSample rs = evaluateRockFacets(vRestPos * mat.noise_scale, nRest, mat.noise_warp, mat.noise_sharpness);
		vec3 rockWorldN = rotateNormalRestToWorld(rs.normal, nRest, nWorld);

		ao = mix(0.55, 1.0, rs.edgeMask);
		shadingNormal = normalize(mix(nWorld, rockWorldN, mat.procedural_shading.x));
		baseColor.rgb = evaluateCosinePalette(mat, 1.0 - rs.edgeMask);
	}

	outNormal = vec4(shadingNormal, ao); // a = AO

	// vec3 macroGrad3D;
	// vec2 macroValPhase = phasorNoise3DGrad(relPos, 1.0, vec3(1.0, 0, 0), vec3(0.0,1.0,0.0), 0.68, uTimeMs, macroGrad3D);
	// float macroVal = macroValPhase.x / max(length(macroValPhase), 0.0001);

	// baseColor.xyz *= macroGrad3D;
	// baseColor.a = 0.15;
	outAlbedo = baseColor;
	outMaterial = vec4(metallicRoughness.x, metallicRoughness.y, 0.0, 0.0); // metallic, roughness, glint
}
