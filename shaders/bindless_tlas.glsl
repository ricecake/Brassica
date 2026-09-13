#ifndef BRASSICA_BINDLESS_TLAS_GLSL
#define BRASSICA_BINDLESS_TLAS_GLSL

// Split out of bindless.glsl deliberately: declaring accelerationStructureEXT bakes the
// RayQueryKHR SPIR-V capability into the shader regardless of whether it's actually read, and
// vkCreateShaderModule checks declared capabilities against enabled device features
// unconditionally. Only include this file from a shader that genuinely uses ray query
// (deferred.frag, today) -- everything else should just #include "bindless.glsl" and stay
// loadable on a device without VK_KHR_ray_query (MinimalDevice, most test coverage).
#extension GL_EXT_ray_query : enable

layout(set = 0, binding = 5) uniform accelerationStructureEXT uTLAS[];

#endif // BRASSICA_BINDLESS_TLAS_GLSL
