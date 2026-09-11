#ifndef BRASSICA_BINDLESS_GLSL
#define BRASSICA_BINDLESS_GLSL

#extension GL_EXT_nonuniform_qualifier : require

// Mirrors the engine's one bindless descriptor set (Engine::InitGlobalDescriptors,
// PhysicalResourceRegistry::BindlessBindings) -- set 0 for any pipeline that doesn't also need
// FrameUBO (see NodeContext::globalSet's comment for why that merge hasn't happened yet).
//
// Binding 4 (acceleration structures) is deliberately NOT declared here -- see
// bindless_tlas.glsl. Merely declaring accelerationStructureEXT with GL_EXT_ray_query enabled
// bakes the RayQueryKHR SPIR-V capability into the shader's module regardless of whether it's
// ever actually read, and vkCreateShaderModule checks declared capabilities against enabled
// device features unconditionally -- so every shader that included this file used to fail to
// load on a device without VK_KHR_ray_query (MinimalDevice, most notably) even if it only ever
// touched uTextures2D/uSamplers. Only deferred.frag genuinely uses ray query; only it should pay
// for declaring it.
layout(set = 0, binding = 0) uniform texture2D uTextures2D[];
layout(set = 0, binding = 1) uniform texture2DArray uTextureArrays[];
layout(set = 0, binding = 2) uniform sampler uSamplers[];
// Format-qualified aliases sharing binding 3 (storage images) -- legal per spec: each entry's
// real format lives on its own image view, and a shader only ever indexes into the alias whose
// qualifier matches what it actually bound (NodeContext::StorageIndex<K>() on the C++ side).
// Only the rgba32f alias exists so far (the two atmosphere LUTs); add a sibling here rather than
// a new binding if a different format shows up later.
layout(set = 0, binding = 3, rgba32f) uniform image2D uImagesRGBA32F[];

// Sampler catalog indices, written once by Engine::InitGlobalDescriptors and injected here via
// Shader::RegisterConstant's [[NAME]] substitution -- GLSL and C++ read the same catalog by
// construction, never by convention.
#define BRASSICA_SAMPLER_NEAREST_CLAMP [[BRASSICA_SAMPLER_NEAREST_CLAMP]]
#define BRASSICA_SAMPLER_LINEAR_CLAMP [[BRASSICA_SAMPLER_LINEAR_CLAMP]]
#define BRASSICA_SAMPLER_LINEAR_REPEAT_MIP [[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]]
#define BRASSICA_SAMPLER_NEAREST_REPEAT [[BRASSICA_SAMPLER_NEAREST_REPEAT]]

// idx is a bindless index into uTextures2D/uTextureArrays (NodeContext::Index<K>() on the C++
// side) -- always wrapped in nonuniformEXT here, once, rather than trusting every call site to
// remember it. A missing/never-registered resource resolves to index 0, the permanent 1x1
// fallback texture (PhysicalRegistry::EnsureFallbackTexture) -- reading it is always well-defined
// visually-wrong, never undefined behavior. NOTE: that fallback is only written into the *plain*
// sampled arena (binding 0) -- SAMPLE_ARRAY_WRAP's binding 1 has no equivalent fallback at index
// 0 yet, so it depends on the caller having actually registered whatever it indexes (true for
// every current SAMPLE_ARRAY_WRAP call site: the terrain clipmap is always registered once at
// Engine::Init before any frame runs).
#define SAMPLE_NEAREST(idx, uv) \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), uv)

#define SAMPLE_LINEAR(idx, uv) \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv)

#define SAMPLE_ARRAY_WRAP(idx, uvw) \
	texture(sampler2DArray(uTextureArrays[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uvw)

#endif // BRASSICA_BINDLESS_GLSL
