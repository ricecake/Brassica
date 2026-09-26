#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <filesystem>
#include <iostream>
#include <string>

#include "doctest/doctest.h"
#include "vulkan/vulkan.hpp"

#include "Shader.hpp"

namespace fs = std::filesystem;

TEST_CASE("Validate All GLSL Shaders Compile Successfully") {
	brassica::Shader::ClearConstants();
	brassica::Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_CLAMP", 0u);
	brassica::Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_CLAMP", 1u);
	brassica::Shader::RegisterConstant("BRASSICA_SAMPLER_LINEAR_REPEAT_MIP", 2u);
	brassica::Shader::RegisterConstant("BRASSICA_SAMPLER_NEAREST_REPEAT", 3u);

	std::string shaderDir = "shaders";
	if (!fs::exists(shaderDir)) {
		if (fs::exists("bin/shaders")) {
			shaderDir = "bin/shaders";
		} else if (fs::exists(std::string(BRASSICA_BUILD_DIR) + "/bin/shaders")) {
			shaderDir = std::string(BRASSICA_BUILD_DIR) + "/bin/shaders";
		} else if (fs::exists(std::string(BRASSICA_BUILD_DIR) + "/../shaders")) {
			shaderDir = std::string(BRASSICA_BUILD_DIR) + "/../shaders";
		}
	}

	if (!fs::exists(shaderDir)) {
		INFO("Shader directory not found: " << shaderDir);
		REQUIRE(false);
	}

	size_t compiledCount = 0;

	for (const auto& entry : fs::recursive_directory_iterator(shaderDir)) {
		if (!entry.is_regular_file()) {
			continue;
		}

		fs::path    path = entry.path();
		std::string ext = path.extension().string();

		if (ext == ".glsl") {
			// Included GLSL helper header -- validated when included by top-level shaders
			continue;
		}

		shaderc_shader_kind kind = shaderc_glsl_infer_from_source;
		if (ext == ".vert") {
			kind = shaderc_glsl_vertex_shader;
		} else if (ext == ".frag") {
			kind = shaderc_glsl_fragment_shader;
		} else if (ext == ".geom") {
			kind = shaderc_glsl_geometry_shader;
		} else if (ext == ".comp") {
			kind = shaderc_glsl_compute_shader;
		} else if (ext == ".tcs" || ext == ".tesc") {
			kind = shaderc_glsl_tess_control_shader;
		} else if (ext == ".tes" || ext == ".tese") {
			kind = shaderc_glsl_tess_evaluation_shader;
		} else if (ext == ".mesh") {
			kind = shaderc_glsl_mesh_shader;
		} else if (ext == ".task") {
			kind = shaderc_glsl_task_shader;
		} else {
			continue;
		}

		brassica::Shader shader;
		bool success = shader.CompileFromFile(vk::Device(nullptr), path.string(), kind);
		INFO("Shader compilation failed for file: " << path.string());
		CHECK(success);

		if (success) {
			++compiledCount;
		}
	}

	MESSAGE("Validated and compiled " << compiledCount << " GLSL shaders successfully.");
	CHECK(compiledCount > 0);

	brassica::Shader::ClearConstants();
}
