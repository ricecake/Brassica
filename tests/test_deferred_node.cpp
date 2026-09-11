#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Shader.hpp"

using namespace brassica;

// GLSL -> SPIR-V compilation (shaderc) is entirely CPU-side and needs no Vulkan device at all --
// passing a null vk::Device skips the (otherwise real, device-capability-checked)
// vkCreateShaderModule call, which is exactly what's wanted here: this test's only job is
// catching a GLSL syntax/semantic error in deferred.frag's new bindless-macro rewrite, not
// proving the module is loadable on any particular device (deferred.frag's real SPIR-V declares
// SPV_KHR_ray_query, so vkCreateShaderModule would need a device with rayQuery enabled --
// MinimalDevice deliberately doesn't have one, and this doesn't need one either).
TEST_CASE("deferred.vert/deferred.frag compile to valid SPIR-V against the real bindless.glsl substitution") {
	// bindless.glsl's BRASSICA_SAMPLER_* macros need these registered before compiling, exactly
	// as Engine::InitGlobalDescriptors does for real -- unregistered, the shader would compile
	// with the literal "[[BRASSICA_SAMPLER_NEAREST_CLAMP]]" tokens still in the source and fail.
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

	VertexShader   vert;
	FragmentShader frag;
	CHECK(vert.CompileVertexFromFile(vk::Device{}, "shaders/deferred.vert"));
	CHECK(frag.CompileFragmentFromFile(vk::Device{}, "shaders/deferred.frag"));
	CHECK(!vert.GetSPIRV().empty());
	CHECK(!frag.GetSPIRV().empty());

	Shader::ClearConstants();
}
