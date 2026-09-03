#include "Shader.hpp"

#include <fstream>
#include <sstream>
#include "spdlog/spdlog.h"

namespace brassica {

	bool Shader::LoadFromFile(const std::string& filepath) {
		this->filePath = filepath;
		std::ifstream file(filepath, std::ios::in | std::ios::binary);
		if (!file.is_open()) {
			spdlog::error("Failed to open shader file: {}", filepath);
			return false;
		}

		std::stringstream buffer;
		buffer << file.rdbuf();
		sourceCode = buffer.str();
		return true;
	}

	bool Shader::CompileFromSource(VkDevice device, const std::string& source, shaderc_shader_kind kind, const char* name) {
		sourceCode = source;

		shaderc::Compiler       compiler;
		shaderc::CompileOptions options;
		options.SetOptimizationLevel(shaderc_optimization_level_performance);

		auto result = compiler.CompileGlslToSpv(source, kind, name, options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
			spdlog::error("Shader compilation error ({}) : {}", name, result.GetErrorMessage());
			return false;
		}

		spirvCode.assign(result.cbegin(), result.cend());

		VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
		createInfo.pCode = spirvCode.data();

		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			spdlog::error("Failed to create Vulkan shader module for {}", name);
			return false;
		}

		return true;
	}

	bool Shader::CompileFromFile(VkDevice device, const std::string& filepath, shaderc_shader_kind kind) {
		if (!LoadFromFile(filepath)) {
			return false;
		}
		return CompileFromSource(device, sourceCode, kind, filepath.c_str());
	}

	void Shader::Destroy(VkDevice device) {
		if (shaderModule != VK_NULL_HANDLE) {
			vkDestroyShaderModule(device, shaderModule, nullptr);
			shaderModule = VK_NULL_HANDLE;
		}
	}

	VkPipelineShaderStageCreateInfo Shader::GetStageCreateInfo(VkShaderStageFlagBits stage) const {
		VkPipelineShaderStageCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		createInfo.stage = stage;
		createInfo.module = shaderModule;
		createInfo.pName = "main";
		return createInfo;
	}

	// ComputeShader
	bool ComputeShader::CompileComputeFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_compute_shader);
	}

	VkPipelineShaderStageCreateInfo ComputeShader::GetStageCreateInfo() const {
		return Shader::GetStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT);
	}

	// GraphicsShader
	VkPipelineShaderStageCreateInfo GraphicsShader::GetStageCreateInfo() const {
		return Shader::GetStageCreateInfo(GetStageFlag());
	}

	// VertexShader
	bool VertexShader::CompileVertexFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_vertex_shader);
	}

	// FragmentShader
	bool FragmentShader::CompileFragmentFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_fragment_shader);
	}

	// GeometryShader
	bool GeometryShader::CompileGeometryFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_geometry_shader);
	}

	// TessControlShader
	bool TessControlShader::CompileTessControlFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_tess_control_shader);
	}

	// TessEvaluationShader
	bool TessEvaluationShader::CompileTessEvalFromFile(VkDevice device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_tess_evaluation_shader);
	}

} // namespace brassica
