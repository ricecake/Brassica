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
	mat4  uPreviousViewProjMatrix;
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
// Formatless *write* aliases of the same binding 3 storage-image catalog, for a resource whose
// real format isn't R32G32B32A32_SFLOAT. Vulkan requires a storage image's declared SPIR-V
// format to match its bound VkImageView's real format unless the shader declares it with no
// format qualifier at all -- gated on shaderStorageImageWriteWithoutFormat (Engine::InitVulkan's
// features1), enabled. Confirmed via real hardware validation: prior to this, every non-RGBA32F
// storage image written through uImagesRGBA32F[] above (most of them -- CloudPackedColor et al.
// are R16G16B16A16_SFLOAT, CloudShadowMap was R16_SFLOAT, etc.) produced a "Format operand
// Rgba32f ... doesn't match the VkImageView format ... undefined values" validation warning on
// every access. AssignAndWriteBindlessIndices/WriteStorageImageDescriptor (PhysicalRegistry.hpp)
// don't care which of these a resource's index ends up written through -- they just write
// whatever real VkImageView the resource has, so switching an imageStore call site from
// uImagesRGBA32F to one of these needs no C++-side change, only a different GLSL declaration
// name at the call site.
//
// writeonly, deliberately -- not a bidirectional or readonly declaration. Confirmed via a real
// glslang compile error ("image variables not declared 'writeonly' and without a format layout
// qualifier ... not supported") that this glslang/shaderc version's GLSL frontend only accepts
// formatless for a writeonly image, full stop -- shaderStorageImageReadWithoutFormat being an
// enabled *device* feature doesn't mean the *language* lets you declare a formatless readonly or
// read-write image; every readonly variant tried failed to compile regardless. Not a problem in
// practice: every current cloud resource with a mismatched format is only ever written via
// imageStore -- everything that reads one back does so through the separate *sampled* catalog
// (SAMPLE_LINEAR/SAMPLE_NEAREST/texture2D, binding 0) rather than imageLoad, so a formatless read
// alias has no real caller yet. If a future resource genuinely needs imageLoad on a formatless
// image, that needs its own investigation (a different Vulkan/SPIR-V target version, an
// explicit per-resource format hint, or restructuring the read through the sampled catalog
// instead) -- don't assume this same writeonly-only fix covers it.
//
// Same alias-the-same-binding idiom terrain_gen.comp (image2DArray) and
// cloud_3d_volume_bake.comp (image3D) already use for *dimensionality* -- centralized here for
// discoverability instead of each shader re-declaring its own local alias, which is exactly how
// a real cascaded-shadow-map port went wrong: the array-alias convention already existed, but
// nothing about it was visible enough for that shader's author to find.
layout(set = 1, binding = 3) writeonly uniform image2D uImagesGenericWrite[];
layout(set = 1, binding = 3) writeonly uniform image2DArray uImageArraysGenericWrite[];
layout(set = 1, binding = 3) writeonly uniform image3D uImages3DGenericWrite[];
// A real, format-qualified, coherent, read-write alias of the same binding 3 catalog -- for the
// one case the writeonly formatless aliases above can't cover: a shader that needs to imageLoad
// its own earlier imageStore within the same dispatch. FidelityFX SPD's single-surviving-
// workgroup "coarse tail" (shaders/terrain_downsample.comp) does exactly this at its mip-6
// handoff -- the elected workgroup reads back what every other workgroup just wrote, so the
// value must actually round-trip, which a formatless declaration can't do here (this glslang
// version only accepts formatless for writeonly, see the comment above). rg32f specifically
// because the only resource that needs this today -- the terrain min/max mip chain -- is RG32F;
// a different format needing this same capability would get its own aliased declaration, same
// pattern as the rest of this file.
layout(set = 1, binding = 3, rg32f) coherent uniform image2D uImagesRG32FCoherent[];
layout(set = 1, binding = 3, rg32f) coherent uniform image2DArray uImageArraysRG32FCoherent[];
// Binding 5: a real 3D-volume sampled catalog, independently sized/counted from bindings 0/1
// (PhysicalRegistry::AssignAndWriteBindlessIndices, sampledImage3DBinding) -- a genuine texture3D,
// not a 2D-array pressed into service for a volume it was never shaped for.
layout(set = 1, binding = 5) uniform texture3D uTextures3D[];

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
#define SAMPLE_NEAREST(idx, uv)                                                                                        \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), uv)

#define SAMPLE_LINEAR(idx, uv)                                                                                         \
	texture(sampler2D(uTextures2D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), uv)

#define SAMPLE_ARRAY_WRAP(idx, uvw)                                                                                    \
	texture(sampler2DArray(uTextureArrays[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uvw)

// idx is a bindless index into uTextures3D (NodeContext::Index<K>() on the C++ side, same as
// SAMPLE_NEAREST/SAMPLE_LINEAR -- StorageIndex<K>() is a different catalog entirely, for
// imageStore/imageLoad writers, not this). Same no-fallback-at-0 caveat as SAMPLE_ARRAY_WRAP:
// depends on the caller having actually registered whatever it indexes.
#define SAMPLE_3D_LINEAR(idx, uvw)                                                                                    \
	texture(sampler3D(uTextures3D[nonuniformEXT(idx)], uSamplers[BRASSICA_SAMPLER_LINEAR_REPEAT_MIP]), uvw)

#endif // BRASSICA_BINDLESS_GLSL
