#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "Shader.hpp"
#include "passes/ScreenSpaceEffectsNode.hpp"
#include "types/ScreenSpacePushConstants.hpp"

using namespace brassica;

TEST_CASE("ScreenSpacePushConstants Layout and Size") {
	CHECK(sizeof(ScreenSpacePushConstants) == 84);
	ScreenSpacePushConstants push{};
	CHECK(push.ssgiEnabled == 1);
	CHECK(push.gtaoEnabled == 1);
	CHECK(push.sssEnabled == 1);
}

TEST_CASE("ScreenSpaceEffectsNode Setup and Recipe Creation") {
	ScreenSpaceEffectsNode node;
	graph::FrameContext ctx{
		.width = 1920,
		.height = 1080,
		.renderScalePercent = 100,
		.frameIndex = 0,
	};

	graph::Recipe r = node.Setup(ctx);
	CHECK(r.domain == graph::ExecutionDomain::Compute);
	CHECK(r.isActive);
	CHECK(r.realizations.size() == 2);
	CHECK(r.realizations[0].key == graph::IdOf<ScreenSpaceIndirectAO>());
	CHECK(r.realizations[1].key == graph::IdOf<ScreenSpaceShadow>());
}

TEST_CASE("shaders/effects/screen_space_effects.comp compiles to valid SPIR-V") {
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
	Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

	ComputeShader comp;
	bool loaded = comp.LoadFromFile("shaders/effects/screen_space_effects.comp");
	CHECK(loaded);
	if (loaded) {
		shaderc::Compiler       compiler;
		shaderc::CompileOptions options;
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
		options.SetTargetSpirv(shaderc_spirv_version_1_5);

		auto res = compiler.CompileGlslToSpv(
			comp.GetSource(),
			shaderc_glsl_compute_shader,
			"screen_space_effects.comp",
			options
		);
		if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
			MESSAGE("screen_space_effects.comp error: ", res.GetErrorMessage());
		}
		CHECK(res.GetCompilationStatus() == shaderc_compilation_status_success);
	}

	Shader::ClearConstants();
}
