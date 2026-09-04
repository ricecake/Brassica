#pragma once

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "shaderc/shaderc.hpp"
#include "vulkan/vulkan.hpp"

namespace brassica {

	class Shader {
	public:
		Shader() = default;
		virtual ~Shader() = default;

		static void RegisterConstant(const std::string& name, const std::string& value);

		static void RegisterConstant(const std::string& name, const char* value) {
			RegisterConstant(name, std::string(value));
		}

		template<typename T>
		static void RegisterConstant(const std::string& name, T value) {
			RegisterConstant(name, std::to_string(value));
		}

		static void ClearConstants();

		static std::unordered_map<std::string, std::string>& GetReplacements();

		bool LoadFromFile(const std::string& filepath);
		bool CompileFromSource(
			vk::Device          device,
			const std::string&  source,
			shaderc_shader_kind kind,
			const char*         name = "Shader"
		);
		bool CompileFromFile(vk::Device device, const std::string& filepath, shaderc_shader_kind kind);
		bool Recompile(vk::Device device);

		void Destroy(vk::Device device);

		vk::ShaderModule GetModule() const { return shaderModule; }

		const std::vector<uint32_t>& GetSPIRV() const { return spirvCode; }

		const std::string& GetSource() const { return sourceCode; }

		const std::string& GetFilePath() const { return filePath; }

		const std::set<std::string>& GetIncludedFiles() const { return includedFiles; }

		shaderc_shader_kind GetKind() const { return shaderKind; }

		virtual vk::PipelineShaderStageCreateInfo GetStageCreateInfo(vk::ShaderStageFlagBits stage) const;

	protected:
		std::string           filePath;
		std::string           sourceCode;
		std::set<std::string> includedFiles;
		std::vector<uint32_t> spirvCode;
		vk::ShaderModule      shaderModule{nullptr};
		shaderc_shader_kind   shaderKind{shaderc_glsl_infer_from_source};
	};

	// Compute Shader Subclass
	class ComputeShader: public Shader {
	public:
		ComputeShader() { shaderKind = shaderc_glsl_compute_shader; }
		bool                              CompileComputeFromFile(vk::Device device, const std::string& filepath);
		vk::PipelineShaderStageCreateInfo GetStageCreateInfo() const;
	};

	// Base class for rendering shader stages
	class GraphicsShader: public Shader {
	public:
		GraphicsShader() = default;
		virtual vk::ShaderStageFlagBits   GetStageFlag() const = 0;
		vk::PipelineShaderStageCreateInfo GetStageCreateInfo() const;
	};

	// Permutations for rendering shader stages
	class VertexShader: public GraphicsShader {
	public:
		VertexShader() { shaderKind = shaderc_glsl_vertex_shader; }
		bool CompileVertexFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eVertex; }
	};

	class FragmentShader: public GraphicsShader {
	public:
		FragmentShader() { shaderKind = shaderc_glsl_fragment_shader; }
		bool CompileFragmentFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eFragment; }
	};

	class GeometryShader: public GraphicsShader {
	public:
		GeometryShader() { shaderKind = shaderc_glsl_geometry_shader; }
		bool CompileGeometryFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eGeometry; }
	};

	class TessControlShader: public GraphicsShader {
	public:
		TessControlShader() { shaderKind = shaderc_glsl_tess_control_shader; }
		bool CompileTessControlFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eTessellationControl; }
	};

	class TessEvaluationShader: public GraphicsShader {
	public:
		TessEvaluationShader() { shaderKind = shaderc_glsl_tess_evaluation_shader; }
		bool CompileTessEvalFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override {
			return vk::ShaderStageFlagBits::eTessellationEvaluation;
		}
	};

	class MeshShader: public GraphicsShader {
	public:
		MeshShader() { shaderKind = shaderc_glsl_mesh_shader; }
		bool CompileMeshFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eMeshEXT; }
	};

	class TaskShader: public GraphicsShader {
	public:
		TaskShader() { shaderKind = shaderc_glsl_task_shader; }
		bool CompileTaskFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eTaskEXT; }
	};

} // namespace brassica
