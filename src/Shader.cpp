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

	bool
	Shader::CompileFromSource(vk::Device device, const std::string& source, shaderc_shader_kind kind, const char* name) {
		sourceCode = source;

		shaderc::Compiler       compiler;
		shaderc::CompileOptions options;
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
		options.SetTargetSpirv(shaderc_spirv_version_1_5);

		auto result = compiler.CompileGlslToSpv(source, kind, name, options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
			spdlog::error("Shader compilation error ({}) : {}", name, result.GetErrorMessage());
			return false;
		}

		spirvCode.assign(result.cbegin(), result.cend());

		vk::ShaderModuleCreateInfo createInfo{};
		createInfo.setCodeSize(spirvCode.size() * sizeof(uint32_t));
		createInfo.setPCode(spirvCode.data());

		try {
			shaderModule = device.createShaderModule(createInfo);
		} catch (const vk::SystemError& err) {
			spdlog::error("Failed to create Vulkan shader module for {}: {}", name, err.what());
			return false;
		}

		return true;
	}

	bool Shader::CompileFromFile(vk::Device device, const std::string& filepath, shaderc_shader_kind kind) {
		if (!LoadFromFile(filepath)) {
			return false;
		}
		return CompileFromSource(device, sourceCode, kind, filepath.c_str());
	}

	void Shader::Destroy(vk::Device device) {
		if (shaderModule) {
			device.destroyShaderModule(shaderModule);
			shaderModule = nullptr;
		}
	}

	vk::PipelineShaderStageCreateInfo Shader::GetStageCreateInfo(vk::ShaderStageFlagBits stage) const {
		vk::PipelineShaderStageCreateInfo createInfo{};
		createInfo.setStage(stage);
		createInfo.setModule(shaderModule);
		createInfo.setPName("main");
		return createInfo;
	}

	// ComputeShader
	bool ComputeShader::CompileComputeFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_compute_shader);
	}

	vk::PipelineShaderStageCreateInfo ComputeShader::GetStageCreateInfo() const {
		return Shader::GetStageCreateInfo(vk::ShaderStageFlagBits::eCompute);
	}

	// GraphicsShader
	vk::PipelineShaderStageCreateInfo GraphicsShader::GetStageCreateInfo() const {
		return Shader::GetStageCreateInfo(GetStageFlag());
	}

	// VertexShader
	bool VertexShader::CompileVertexFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_vertex_shader);
	}

	// FragmentShader
	bool FragmentShader::CompileFragmentFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_fragment_shader);
	}

	// GeometryShader
	bool GeometryShader::CompileGeometryFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_geometry_shader);
	}

	// TessControlShader
	bool TessControlShader::CompileTessControlFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_tess_control_shader);
	}

	// TessEvaluationShader
	bool TessEvaluationShader::CompileTessEvalFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_tess_evaluation_shader);
	}

} // namespace brassica
