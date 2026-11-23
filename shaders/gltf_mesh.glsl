#ifndef BRASSICA_GLTF_MESH_GLSL
#define BRASSICA_GLTF_MESH_GLSL

#include "bindless.glsl"

// GPU-driven static (Stage 1) GLTF mesh data: vertices pre-gathered per meshlet by
// GltfModel's CPU-side loader (fastgltf parse + meshopt_buildMeshlets), not sampled by
// accessor at draw time. JOINTS_0/WEIGHTS_0 are carried on every vertex (zero-filled for
// non-skinned primitives) rather than added later -- skeletal animation is an already-committed
// later stage of this same pipeline, not a hypothetical, so there's one vertex layout instead of
// two that would need reconciling.
//
// Byte offsets verified against the real compiled std430 layout via spirv-dis, not hand-computed
// (see feedback_push_constant_struct_editing) -- GltfVertex is 80 bytes/vertex (vec3 pads to 16
// before the next vec3, but not before a following scalar), GltfMeshlet is 64 bytes/meshlet. Any
// C++ mirror struct (model/GltfTypes.hpp) must match these exactly, including trailing padding so
// sizeof() agrees with the array stride used for indexing.
struct GltfVertex {
	vec3  position;
	vec3  normal;
	vec2  uv;
	uvec4 joints;  // JOINTS_0 -- indices into this model's per-instance joint-matrix array
	vec4  weights; // WEIGHTS_0
};

// Mirrors meshopt_Meshlet (vertex/triangle offset+count into the shared gathered-vertex and
// micro-index-triangle buffers) plus the subset of meshopt_Bounds a task-shader cull actually
// needs (bounding sphere for frustum/occlusion, normal cone for backface) -- see
// meshopt_computeMeshletBounds's own doc comment for the cone-culling formulas. materialIndex is
// per-meshlet, not per-vertex: GltfModel's loader never merges primitives with different
// materials into one meshlet.
struct GltfMeshlet {
	uint  vertexOffset;
	uint  triangleOffset;
	uint  vertexCount;
	uint  triangleCount;
	vec3  boundsCenter;
	float boundsRadius;
	vec3  coneAxis;
	float coneCutoff;
	uint  materialIndex;
};

// Binding 6 aliases (see material_system.glsl's own comment on this binding) -- each buffer is
// eStorageBuffer-usage and already auto-registered by PhysicalRegistry::AssignAndWriteBindlessBufferIndex,
// so ctx.BufferIndex<K>() is all the C++ side needs; no device-address plumbing.
layout(set = 1, binding = 6, std430) readonly buffer GltfVertexBufferArray {
	GltfVertex vertices[];
}
uGltfVertexBuffers[];

// meshlet_vertices from meshopt_buildMeshlets: for each meshlet's local vertex index, which
// *gathered* GltfVertex to use (GltfModel's loader already re-gathers per-meshlet, so this is a
// dense local range, not an index back into some larger un-gathered vertex pool).
layout(set = 1, binding = 6, std430) readonly buffer GltfMeshletVertexBufferArray {
	uint indices[];
}
uGltfMeshletVertexBuffers[];

// meshlet_triangles from meshopt_buildMeshlets is natively an 8-bit micro-index buffer, 3 per
// triangle, local to whichever meshlet owns them -- but reading individual bytes from a buffer in
// GLSL needs GL_EXT_shader_explicit_arithmetic_types_int8, which has no precedent anywhere in
// this codebase and isn't confirmed enabled on the device (no shaderInt8 feature request in
// Engine.cpp). GltfModel's loader repacks 4 micro-indices per uint instead (same bytes, just
// addressed a word at a time): byteIndex = meshlet.triangleOffset*3 + triangleLocal*3 + corner;
// word = indices[byteIndex >> 2]; value = (word >> ((byteIndex & 3) * 8)) & 0xFFu.
layout(set = 1, binding = 6, std430) readonly buffer GltfMeshletTriangleBufferArray {
	uint indices[];
}
uGltfMeshletTriangleBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer GltfMeshletBufferArray {
	GltfMeshlet meshlets[];
}
uGltfMeshletBuffers[];

#endif // BRASSICA_GLTF_MESH_GLSL
