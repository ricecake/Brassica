#ifndef BRASSICA_BINDLESS_GLSL
#define BRASSICA_BINDLESS_GLSL

#extension GL_EXT_nonuniform_qualifier : require

// Frame UBO in its own always-bound set 0 -- every shader gets ready access to camera/time/
// frame data without declaring it as a graph resource dependency. Genuinely double-buffered on
// the C++ side (Engine::frameDescriptorSets), unlike the bindless catalog below, since its
// contents are CPU-written fresh every frame.
layout(std140, set = 0, binding = 0) uniform FrameUBO {
	mat4  uViewMatrix;
	mat4  uInvViewMatrix;
	mat4  uProjMatrix;
	mat4  uInvProjMatrix;
	mat4  uViewProjMatrix;
	mat4  uInvViewProjMatrix;
	vec4  uCameraPosition;
	vec4  uPreviousCameraPosition;
	float uTime;
	float uFov;
	float uAspectRatio;
	float uNearPlane;
	float uFarPlane;
	uint  uFrameIndex;
	uint  uGlobalSeed;
	uint  uFrameRandom;
};

// Bindless resource catalog in Set 1 (bindings 0..4). One descriptor set instance -- never
// duplicated per frame, since a resource's descriptor is written once at creation and read for
// the rest of its life.
//
// Binding 4 (acceleration structures) is deliberately NOT declared here -- see
// bindless_tlas.glsl.
layout(set = 1, binding = 0) uniform texture2D uTextures2D[];
layout(set = 1, binding = 1) uniform texture2DArray uTextureArrays[];
layout(set = 1, binding = 2) uniform sampler uSamplers[];
layout(set = 1, binding = 3, rgba32f) uniform image2D uImagesRGBA32F[];
layout(set = 1, binding = 5) uniform texture3D uTextures3D[];
layout(set = 1, binding = 6, rgba16f) uniform image3D uImages3D[];
layout(set = 1, binding = 7, r16f) uniform image2DArray uImageArrays[];

// Sampler catalog indices, written once by Engine::InitGlobalDescriptors and injected here via
// Shader::RegisterConstant's [[NAME]] substitution -- GLSL and C++ read the same catalog by
// construction, never by convention.
#define BRASSICA_SAMPLER_NEAREST_CLAMP [[BRASSICA_SAMPLER_NEAREST_CLAMP]]
#define BRASSICA_SAMPLER_LINEAR_CLAMP [[BRASSICA_SAMPLER_LINEAR_CLAMP]]
#define BRASSICA_SAMPLER_LINEAR_REPEAT_MIP [[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]]
#define BRASSICA_SAMPLER_NEAREST_REPEAT [[BRASSICA_SAMPLER_NEAREST_REPEAT]]

// idx is a bindless index into uTextures2D/uTextureArrays/uTextures3D (NodeContext::Index<K>() on the C++
// side) -- always wrapped in nonuniformEXT here, once, rather than trusting every call site to
// remember it. A missing/never-registered resource resolves to index 0, the permanent 1x1
// fallback texture (PhysicalRegistry::EnsureFallbackTexture) -- reading it is always well-defined
// visually-wrong, never undefined behavior. NOTE: that fallback is only written into the *plain*
// sampled arena (binding 0) -- SAMPLE_ARRAY_WRAP's binding 1 has no equivalent fallback at index
// 0 yet, so it depends on the caller having actually registered whatever it indexes (true for
// every current SAMPLE_ARRAY_WRAP call site: the terrain clipmap is always registered once at
// Engine::Init before any frame runs).
#define SAMPLE_NEAREST(idx, uv)                                                                                        \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), uv)

#define SAMPLE_LINEAR(idx, uv)                                                                                         \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv)

#define SAMPLE_ARRAY_WRAP(idx, uvw)                                                                                    \
	texture(sampler2DArray(uTextureArrays[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uvw)

#define SAMPLE_3D_NEAREST(idx, uvw)                                                                                    \
	texture(sampler3D(uTextures3D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), uvw)

#define SAMPLE_3D_LINEAR(idx, uvw)                                                                                     \
	texture(sampler3D(uTextures3D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uvw)

#define SAMPLE_3D_REPEAT_MIP(idx, uvw)                                                                                 \
	texture(sampler3D(uTextures3D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uvw)

#define SAMPLE_3D_LOD(idx, uvw, lod)                                                                                   \
	textureLod(sampler3D(uTextures3D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uvw, lod)

#define SAMPLE_ARRAY_LINEAR_LOD(idx, uvw, lod)                                                                         \
	textureLod(sampler2DArray(uTextureArrays[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uvw, lod)

#endif // BRASSICA_BINDLESS_GLSL
