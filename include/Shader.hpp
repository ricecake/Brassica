#pragma once

#include <string>
#include <vector>

#include "shaderc/shaderc.hpp"
#include "vulkan/vulkan.hpp"

namespace brassica {

	class Shader {
	public:
		Shader() = default;
		virtual ~Shader() = default;

		bool LoadFromFile(const std::string& filepath);
		bool CompileFromSource(
			vk::Device          device,
			const std::string&  source,
			shaderc_shader_kind kind,
			const char*         name = "Shader"
		);
		bool CompileFromFile(vk::Device device, const std::string& filepath, shaderc_shader_kind kind);

		void Destroy(vk::Device device);

		vk::ShaderModule GetModule() const { return shaderModule; }

		const std::vector<uint32_t>& GetSPIRV() const { return spirvCode; }

		const std::string& GetSource() const { return sourceCode; }

		const std::string& GetFilePath() const { return filePath; }

		virtual vk::PipelineShaderStageCreateInfo GetStageCreateInfo(vk::ShaderStageFlagBits stage) const;

	protected:
		std::string           filePath;
		std::string           sourceCode;
		std::vector<uint32_t> spirvCode;
		vk::ShaderModule      shaderModule{nullptr};
	};

	// Compute Shader Subclass
	class ComputeShader: public Shader {
	public:
		ComputeShader() = default;
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
		VertexShader() = default;
		bool CompileVertexFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eVertex; }
	};

	class FragmentShader: public GraphicsShader {
	public:
		FragmentShader() = default;
		bool CompileFragmentFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eFragment; }
	};

	class GeometryShader: public GraphicsShader {
	public:
		GeometryShader() = default;
		bool CompileGeometryFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eGeometry; }
	};

	class TessControlShader: public GraphicsShader {
	public:
		TessControlShader() = default;
		bool CompileTessControlFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eTessellationControl; }
	};

	class TessEvaluationShader: public GraphicsShader {
	public:
		TessEvaluationShader() = default;
		bool CompileTessEvalFromFile(vk::Device device, const std::string& filepath);

		vk::ShaderStageFlagBits GetStageFlag() const override { return vk::ShaderStageFlagBits::eTessellationEvaluation; }
	};

} // namespace brassica
