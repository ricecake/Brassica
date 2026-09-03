#pragma once

#include <string>
#include <vector>

#include "shaderc/shaderc.hpp"
#include "vulkan/vulkan.h"

namespace brassica {

	class Shader {
	public:
		Shader() = default;
		virtual ~Shader() = default;

		bool LoadFromFile(const std::string& filepath);
		bool CompileFromSource(
			VkDevice            device,
			const std::string&  source,
			shaderc_shader_kind kind,
			const char*         name = "Shader"
		);
		bool CompileFromFile(VkDevice device, const std::string& filepath, shaderc_shader_kind kind);

		void Destroy(VkDevice device);

		VkShaderModule GetModule() const { return shaderModule; }

		const std::vector<uint32_t>& GetSPIRV() const { return spirvCode; }

		const std::string& GetSource() const { return sourceCode; }

		const std::string& GetFilePath() const { return filePath; }

		virtual VkPipelineShaderStageCreateInfo GetStageCreateInfo(VkShaderStageFlagBits stage) const;

	protected:
		std::string           filePath;
		std::string           sourceCode;
		std::vector<uint32_t> spirvCode;
		VkShaderModule        shaderModule{VK_NULL_HANDLE};
	};

	// Compute Shader Subclass
	class ComputeShader: public Shader {
	public:
		ComputeShader() = default;
		bool                            CompileComputeFromFile(VkDevice device, const std::string& filepath);
		VkPipelineShaderStageCreateInfo GetStageCreateInfo() const;
	};

	// Base class for rendering shader stages
	class GraphicsShader: public Shader {
	public:
		GraphicsShader() = default;
		virtual VkShaderStageFlagBits   GetStageFlag() const = 0;
		VkPipelineShaderStageCreateInfo GetStageCreateInfo() const;
	};

	// Permutations for rendering shader stages
	class VertexShader: public GraphicsShader {
	public:
		VertexShader() = default;
		bool CompileVertexFromFile(VkDevice device, const std::string& filepath);

		VkShaderStageFlagBits GetStageFlag() const override { return VK_SHADER_STAGE_VERTEX_BIT; }
	};

	class FragmentShader: public GraphicsShader {
	public:
		FragmentShader() = default;
		bool CompileFragmentFromFile(VkDevice device, const std::string& filepath);

		VkShaderStageFlagBits GetStageFlag() const override { return VK_SHADER_STAGE_FRAGMENT_BIT; }
	};

	class GeometryShader: public GraphicsShader {
	public:
		GeometryShader() = default;
		bool CompileGeometryFromFile(VkDevice device, const std::string& filepath);

		VkShaderStageFlagBits GetStageFlag() const override { return VK_SHADER_STAGE_GEOMETRY_BIT; }
	};

	class TessControlShader: public GraphicsShader {
	public:
		TessControlShader() = default;
		bool CompileTessControlFromFile(VkDevice device, const std::string& filepath);

		VkShaderStageFlagBits GetStageFlag() const override { return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; }
	};

	class TessEvaluationShader: public GraphicsShader {
	public:
		TessEvaluationShader() = default;
		bool CompileTessEvalFromFile(VkDevice device, const std::string& filepath);

		VkShaderStageFlagBits GetStageFlag() const override { return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; }
	};

} // namespace brassica
