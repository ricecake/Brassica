#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "passes/TerrainNode.hpp"
#include "passes/WaterNode.hpp"
#include "Shader.hpp"

using namespace brassica;

namespace {

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

TEST_CASE("TerrainPushConstants is exactly as large as terrain.task/terrain.mesh's shared push_constant block") {
	CHECK(sizeof(TerrainPushConstants) == 80);
}

TEST_CASE("water.task/water.mesh/water.frag compile to valid SPIR-V against the real bindless.glsl substitution") {
	RegisterBindlessSamplerConstants();

	TaskShader     task;
	MeshShader     mesh;
	FragmentShader frag;
	CHECK(task.CompileTaskFromFile(vk::Device{}, "shaders/water.task"));
	CHECK(mesh.CompileMeshFromFile(vk::Device{}, "shaders/water.mesh"));
	CHECK(frag.CompileFragmentFromFile(vk::Device{}, "shaders/water.frag"));
	CHECK(!task.GetSPIRV().empty());
	CHECK(!mesh.GetSPIRV().empty());
	CHECK(!frag.GetSPIRV().empty());

	Shader::ClearConstants();
}

TEST_CASE("WaterPushConstants is exactly as large as water.task/water.mesh/water.frag's shared push_constant block") {
	CHECK(sizeof(WaterPushConstants) == 48);
}
