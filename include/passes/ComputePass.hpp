#pragma once

#include <span>
#include "passes/Pass.hpp"
#include "Shader.hpp"

namespace brassica {

	class ShaderWatcher;

	class ComputePass : public Pass {
	public:
		ComputePass(std::string name, vk::Device device);
		~ComputePass() override;

		void SetShader(ComputeShader* shader);

		void InitComputePipeline(
			std::span<const vk::DescriptorSetLayout> setLayouts = {},
			std::span<const vk::PushConstantRange>   pushConstants = {},
			ShaderWatcher*                          watcher = nullptr
		);

		void Dispatch(vk::CommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1) const;

	protected:
		std::vector<vk::DescriptorSetLayout> storedSetLayouts;
		std::vector<vk::PushConstantRange>   storedPushConstants;

		ComputeShader* computeShader{nullptr};
	};

} // namespace brassica
