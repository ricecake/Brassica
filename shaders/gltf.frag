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

	// vWorldPos is already camera-relative (see gltf_render.mesh) -- matches every other GBuffer
	// writer's outPosition convention (terrain.frag/ball.frag/cylinder.frag/foliage.frag all store
	// position relative to camera, not absolute world position).
	outPosition = vec4(vWorldPos, 0.0); // w = emissivity; 0.0 matches every non-foliage GBuffer
	                                     // writer (see foliage.frag's own emissivity fix this session).
	                                     // Real additive glTF emissive isn't composited here yet --
	                                     // see material_system.glsl's own comment on why.
	outNormal = vec4(normalize(vNormal), occlusion); // a = AO
	outAlbedo = baseColor;
	outMaterial = vec4(metallicRoughness.x, metallicRoughness.y, 0.0, 0.0); // metallic, roughness, glint
}
