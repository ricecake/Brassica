#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"
#include "Shader.hpp"

using namespace brassica;

// GLSL -> SPIR-V compilation (shaderc) is entirely CPU-side and needs no Vulkan device at all --
// passing a null vk::Device skips the (otherwise real, device-capability-checked)
// vkCreateShaderModule call, which is exactly what's wanted here: these tests' only job is
// catching a GLSL syntax/semantic error in a real ported shader's bindless-macro rewrite, not
// proving the module is loadable on any particular device (deferred.frag's real SPIR-V declares
// SPV_KHR_ray_query, so vkCreateShaderModule would need a device with rayQuery enabled --
// MinimalDevice deliberately doesn't have one, and none of these tests need one either).
namespace {

	// bindless.glsl's BRASSICA_SAMPLER_* macros need these registered before compiling, exactly
	// as Engine::InitGlobalDescriptors does for real -- unregistered, a shader that includes
	// bindless.glsl would compile with the literal "[[BRASSICA_SAMPLER_NEAREST_CLAMP]]" tokens
	// still in the source and fail.
	void RegisterBindlessSamplerConstants() {
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
		Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);
	}

} // namespace

TEST_CASE("deferred.vert/deferred.frag compile to valid SPIR-V against the real bindless.glsl substitution") {
	RegisterBindlessSamplerConstants();

	VertexShader   vert;
	FragmentShader frag;
	CHECK(vert.CompileVertexFromFile(vk::Device{}, "shaders/deferred.vert"));
	CHECK(frag.CompileFragmentFromFile(vk::Device{}, "shaders/deferred.frag"));
	CHECK(!vert.GetSPIRV().empty());
	CHECK(!frag.GetSPIRV().empty());

	Shader::ClearConstants();
}

// terrain.task deliberately doesn't include bindless.glsl (it never samples the clipmap, only
// terrain.mesh does), so registering the constants here is only exercised by the mesh shader --
// harmless either way, since RegisterConstant's substitution is a no-op for a shader whose
// source never contains the [[NAME]] placeholder.
TEST_CASE("terrain.task/terrain.mesh compile to valid SPIR-V against the real bindless.glsl substitution") {
	RegisterBindlessSamplerConstants();

	TaskShader task;
	MeshShader mesh;
	CHECK(task.CompileTaskFromFile(vk::Device{}, "shaders/terrain.task"));
	CHECK(mesh.CompileMeshFromFile(vk::Device{}, "shaders/terrain.mesh"));
	CHECK(!task.GetSPIRV().empty());
	CHECK(!mesh.GetSPIRV().empty());

	Shader::ClearConstants();
}

// Regression check for a real failure caught on real hardware mid-port: terrain.mesh's
// push_constant block must be exactly as large as TerrainPushConstants, or
// vkCreateGraphicsPipelines rejects the pipeline ("push constant buffer Block with range [0, N]
// which outside the VkPushConstantRange"). Pins the exact byte count so a future drift between
// the C++ struct and the GLSL block fails loudly here instead of only on someone's GPU.
TEST_CASE("TerrainPushConstants is exactly as large as terrain.task/terrain.mesh's shared push_constant block") {
	CHECK(sizeof(TerrainPushConstants) == 52);
}

TEST_CASE("water.mesh/water.frag compile to valid SPIR-V against the real bindless.glsl substitution") {
	RegisterBindlessSamplerConstants();

	MeshShader     mesh;
	FragmentShader frag;
	CHECK(mesh.CompileMeshFromFile(vk::Device{}, "shaders/water.mesh"));
	CHECK(frag.CompileFragmentFromFile(vk::Device{}, "shaders/water.frag"));
	CHECK(!mesh.GetSPIRV().empty());
	CHECK(!frag.GetSPIRV().empty());

	Shader::ClearConstants();
}

TEST_CASE("WaterPushConstants is exactly as large as water.mesh/water.frag's shared push_constant block") {
	CHECK(sizeof(WaterPushConstants) == 28);
}
