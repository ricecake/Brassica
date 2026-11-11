#version 460

#include "helpers/material_system.glsl"

layout(location = 0) in vec3      vWorldPos;
layout(location = 1) in vec3      vNormal;
layout(location = 2) in vec2      vUv;
layout(location = 3) flat in uint vMaterialIndex;
layout(location = 4) flat in uint vMaterialBufferIndex;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

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
	outNormal = vec4(normalize(vNormal), occlusion); // a = AO
	outAlbedo = baseColor;
	outMaterial = vec4(metallicRoughness.x, metallicRoughness.y, 0.0, 0.0); // metallic, roughness, glint
}
