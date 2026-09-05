#include "Shader.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "spdlog/spdlog.h"

namespace brassica {

	namespace {

		std::string normalizePath(const std::string& path) {
			namespace fs = std::filesystem;
			try {
				fs::path p(path);
				std::string normalized = fs::weakly_canonical(p).string();
				std::replace(normalized.begin(), normalized.end(), '\\', '/');
				return normalized;
			} catch (...) {
				std::string normalized = path;
				std::replace(normalized.begin(), normalized.end(), '\\', '/');
				return normalized;
			}
		}

		std::string generateIncludeGuard(const std::string& path) {
			std::string guard = "G";
			for (char c : path) {
				if (std::isalnum(static_cast<unsigned char>(c))) {
					guard += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
				} else if (!guard.empty() && guard.back() != '_') {
					guard += '_';
				}
			}
			while (!guard.empty() && guard.back() == '_') {
				guard.pop_back();
			}
			return guard;
		}

		std::string loadFileRaw(const std::string& path) {
			std::ifstream file(path, std::ios::in | std::ios::binary);
			if (file.is_open()) {
				std::stringstream buffer;
				buffer << file.rdbuf();
				return buffer.str();
			}

			// Try fallback locations if relative path wasn't found directly
			std::vector<std::string> fallbacks = {
				"bin/" + path,
				std::string(BRASSICA_BUILD_DIR) + "/bin/" + path,
				std::string(BRASSICA_BUILD_DIR) + "/" + path,
				std::string(BRASSICA_BUILD_DIR) + "/../" + path
			};

			for (const auto& fb : fallbacks) {
				std::ifstream fbFile(fb, std::ios::in | std::ios::binary);
				if (fbFile.is_open()) {
					std::stringstream buffer;
					buffer << fbFile.rdbuf();
					return buffer.str();
				}
			}

			return "";
		}

		std::string loadShaderSourceInternal(
			const std::string& path,
			std::set<std::string>& includedFiles,
			const std::string& stageDefine = ""
		) {
			namespace fs = std::filesystem;
			fs::path p(path);
			std::string normalizedPath = normalizePath(path);

			bool isTopLevel = includedFiles.empty();

			if (includedFiles.count(normalizedPath)) {
				// Prevent circular inclusion and ensure single inclusion
				return "";
			}

			std::string sourceCode = loadFileRaw(normalizedPath);
			if (sourceCode.empty() && !loadFileRaw(path).empty()) {
				sourceCode = loadFileRaw(path);
			}

			if (sourceCode.empty()) {
				if (!isTopLevel) {
					spdlog::error("Shader include file not found or empty: {}", path);
				}
				return "";
			}

			includedFiles.insert(normalizedPath);

			std::string guard = generateIncludeGuard(normalizedPath);
			std::string versionLine;
			std::istringstream iss(sourceCode);
			std::string line;

			std::string preVersionContent;
			std::string postVersionContent;
			bool foundVersion = false;

			while (std::getline(iss, line)) {
				std::string trimmed = line;
				size_t firstNonWhitespace = trimmed.find_first_not_of(" \t\r\n");
				if (firstNonWhitespace != std::string::npos) {
					trimmed.erase(0, firstNonWhitespace);
				}

				if (trimmed.substr(0, 8) == "#include") {
					size_t firstQuote = line.find('"');
					size_t lastQuote = line.rfind('"');
					if (firstQuote != std::string::npos && lastQuote != std::string::npos && firstQuote < lastQuote) {
						std::string includePath = line.substr(firstQuote + 1, lastQuote - firstQuote - 1);

						std::vector<fs::path> searchPaths;
						if (includePath != path) {
							searchPaths.push_back(p.parent_path() / includePath);
						}
						searchPaths.push_back(fs::path("shaders") / includePath);
						searchPaths.push_back(fs::path("external") / includePath);
						searchPaths.push_back(fs::path(includePath));

						std::string fullPathStr = "";
						for (const auto& candidate : searchPaths) {
							if (fs::exists(candidate) && !fs::is_directory(candidate)) {
								fullPathStr = candidate.string();
								break;
							}
						}

						if (!fullPathStr.empty()) {
							std::string includedSource = loadShaderSourceInternal(fullPathStr, includedFiles);
							std::string commentStart = "//START " + fullPathStr + "\n";
							std::string commentEnd = "//END " + fullPathStr + " (returning to " + normalizedPath + ")\n";

							if (foundVersion) {
								postVersionContent += commentStart + includedSource;
								if (!includedSource.empty() && includedSource.back() != '\n')
									postVersionContent += "\n";
								postVersionContent += commentEnd;
							} else {
								preVersionContent += commentStart + includedSource;
								if (!includedSource.empty() && includedSource.back() != '\n')
									preVersionContent += "\n";
								preVersionContent += commentEnd;
							}
						} else {
							spdlog::error("Shader #include error: file not found for {}", includePath);
						}
						continue;
					}
				}

				if (!foundVersion) {
					if (trimmed.substr(0, 8) == "#version") {
						versionLine = line + "\n";
						foundVersion = true;
						continue;
					}
					preVersionContent += line + "\n";
				} else {
					postVersionContent += line + "\n";
				}
			}

			if (!foundVersion) {
				postVersionContent = preVersionContent;
				preVersionContent = "";
			}

			std::string shaderStageDefine = stageDefine;
			if (isTopLevel && shaderStageDefine.empty()) {
				std::string ext = p.extension().string();
				if (ext == ".vert") shaderStageDefine = "VERTEX_SHADER";
				else if (ext == ".frag") shaderStageDefine = "FRAGMENT_SHADER";
				else if (ext == ".geom") shaderStageDefine = "GEOMETRY_SHADER";
				else if (ext == ".comp") shaderStageDefine = "COMPUTE_SHADER";
				else if (ext == ".tcs" || ext == ".tesc") shaderStageDefine = "TESS_CONTROL_SHADER";
				else if (ext == ".tes" || ext == ".tese") shaderStageDefine = "TESS_EVALUATION_SHADER";
				else if (ext == ".mesh") shaderStageDefine = "MESH_SHADER";
				else if (ext == ".task") shaderStageDefine = "TASK_SHADER";
			}

			std::string finalSource = versionLine;
			if (isTopLevel && !shaderStageDefine.empty()) {
				finalSource += "#define " + shaderStageDefine + "\n";
			}
			finalSource += "#ifndef " + guard + "\n";
			finalSource += "#define " + guard + "\n";
			finalSource += preVersionContent;
			finalSource += postVersionContent;
			finalSource += "#endif // " + guard + "\n";

			for (auto const& [placeholder, value] : Shader::GetReplacements()) {
				size_t pos = 0;
				while ((pos = finalSource.find(placeholder, pos)) != std::string::npos) {
					finalSource.replace(pos, placeholder.length(), value);
					pos += value.length();
				}
			}

			return finalSource;
		}

	} // namespace

	std::unordered_map<std::string, std::string>& Shader::GetReplacements() {
		static std::unordered_map<std::string, std::string> replacements;
		return replacements;
	}

	void Shader::RegisterConstant(const std::string& name, const std::string& value) {
		GetReplacements()["[[" + name + "]]"] = value;
	}

	void Shader::ClearConstants() {
		GetReplacements().clear();
	}

	bool Shader::LoadFromFile(const std::string& filepath) {
		this->filePath = filepath;
		this->includedFiles.clear();

		std::string processedSource = loadShaderSourceInternal(filepath, this->includedFiles);
		if (processedSource.empty()) {
			spdlog::error("Failed to load or process shader file: {}", filepath);
			return false;
		}

		this->sourceCode = processedSource;
		return true;
	}

	bool Shader::CompileFromSource(
		vk::Device          device,
		const std::string&  source,
		shaderc_shader_kind kind,
		const char*         name
	) {
		sourceCode = source;
		shaderKind = kind;

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

		if (device) {
			try {
				shaderModule = device.createShaderModule(createInfo);
			} catch (const vk::SystemError& err) {
				spdlog::error("Failed to create Vulkan shader module for {}: {}", name, err.what());
				return false;
			}
		}

		return true;
	}

	bool Shader::CompileFromFile(vk::Device device, const std::string& filepath, shaderc_shader_kind kind) {
		if (!LoadFromFile(filepath)) {
			return false;
		}
		return CompileFromSource(device, sourceCode, kind, filepath.c_str());
	}

	bool Shader::Recompile(vk::Device device) {
		if (filePath.empty()) {
			spdlog::error("Cannot recompile shader without filepath");
			return false;
		}

		std::set<std::string> newIncludedFiles;
		std::string newSource = loadShaderSourceInternal(filePath, newIncludedFiles);
		if (newSource.empty()) {
			spdlog::error("Failed to load shader file for recompile: {}", filePath);
			return false;
		}

		shaderc::Compiler       compiler;
		shaderc::CompileOptions options;
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
		options.SetTargetSpirv(shaderc_spirv_version_1_6);

		auto result = compiler.CompileGlslToSpv(newSource, shaderKind, filePath.c_str(), options);
		if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
			spdlog::error("Shader re-compilation error ({}) : {}", filePath, result.GetErrorMessage());
			return false;
		}

		std::vector<uint32_t> newSpirv(result.cbegin(), result.cend());

		vk::ShaderModuleCreateInfo createInfo{};
		createInfo.setCodeSize(newSpirv.size() * sizeof(uint32_t));
		createInfo.setPCode(newSpirv.data());

		vk::ShaderModule newModule{nullptr};
		if (device) {
			try {
				newModule = device.createShaderModule(createInfo);
			} catch (const vk::SystemError& err) {
				spdlog::error("Failed to create Vulkan shader module during recompile for {}: {}", filePath, err.what());
				return false;
			}
		}

		Destroy(device);
		sourceCode = std::move(newSource);
		includedFiles = std::move(newIncludedFiles);
		spirvCode = std::move(newSpirv);
		shaderModule = newModule;
		return true;
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

	// MeshShader
	bool MeshShader::CompileMeshFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_mesh_shader);
	}

	// TaskShader
	bool TaskShader::CompileTaskFromFile(vk::Device device, const std::string& filepath) {
		return CompileFromFile(device, filepath, shaderc_glsl_task_shader);
	}

} // namespace brassica
