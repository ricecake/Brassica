#version 460

#include "helpers/material_system.glsl"
#include "common.glsl"

layout(location = 0) in vec3      vWorldPos;
layout(location = 1) in vec3      vNormal;
layout(location = 2) in vec2      vUv;
layout(location = 3) flat in uint vMaterialIndex;
layout(location = 4) flat in uint vMaterialBufferIndex;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

const float WIND_STRENGTH = 0.85; // 0.0 = calm, 1.0 = steady breeze
const float LEAF_SCALE    = 7.5;  // Higher = smaller, denser leaves on the hull

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

// Generates overlapping, randomly oriented 3D leaf blades across a solid hull
LeafSample evaluateLeafCanopy(vec3 p, vec3 macroNormal) {
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
    res.edgeMask   = smoothstep(0.0, 0.14, f2 - f1);
    res.midrib     = 1.0 - smoothstep(0.0, 0.06, abs(bestLocalR.y));
    res.radialDist = clamp(f1, 0.0, 1.0);
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
    LeafSample ls = evaluateLeafCanopy(vWorldPos * LEAF_SCALE, vNormal);
    float ao = mix(0.45, 1.0, ls.edgeMask) * mix(0.65, 1.0, vNormal.y * 0.5 + 0.5);

	outNormal = vec4(normalize(ls.normal), ao); // a = AO

	// vec3 macroGrad3D;
	// vec2 macroValPhase = phasorNoise3DGrad(relPos, 1.0, vec3(1.0, 0, 0), vec3(0.0,1.0,0.0), 0.68, uTimeMs, macroGrad3D);
	// float macroVal = macroValPhase.x / max(length(macroValPhase), 0.0001);

	// baseColor.xyz *= macroGrad3D;
    // baseColor.a = 0.15;
	outAlbedo = baseColor;
	outMaterial = vec4(metallicRoughness.x, metallicRoughness.y, 0.0, 0.0); // metallic, roughness, glint
}
