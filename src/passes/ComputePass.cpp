#include "passes/ComputePass.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	ComputePass::ComputePass(std::string name, vk::Device dev)
		: Pass(std::move(name), dev) {}

	ComputePass::~ComputePass() {
		DestroyPipeline();
	}

	void ComputePass::SetShader(ComputeShader* shader) {
		computeShader = shader;
	}

	void ComputePass::InitComputePipeline(
		std::span<const vk::DescriptorSetLayout> setLayouts,
		std::span<const vk::PushConstantRange>   pushConstants,
		ShaderWatcher*                          watcher
	) {
		auto buildPipeline = [this, setLayouts, pushConstants]() {
			if (!computeShader) {
				spdlog::error("Cannot build ComputePass pipeline for {}: compute shader not set.", name);
				return;
			}

			if (pipeline) {
				device.destroyPipeline(pipeline);
				pipeline = nullptr;
			}
			if (pipelineLayout) {
				device.destroyPipelineLayout(pipelineLayout);
				pipelineLayout = nullptr;
			}

			vk::PipelineLayoutCreateInfo layoutInfo{};
			layoutInfo.setSetLayouts(setLayouts);
			layoutInfo.setPushConstantRanges(pushConstants);

			try {
				pipelineLayout = device.createPipelineLayout(layoutInfo);
			} catch (const vk::SystemError& err) {
				spdlog::error("Failed to create compute pipeline layout for pass {}: {}", name, err.what());
				return;
			}

			vk::ComputePipelineCreateInfo pipelineInfo{};
			pipelineInfo.setStage(computeShader->GetStageCreateInfo());
			pipelineInfo.setLayout(pipelineLayout);

			auto result = device.createComputePipeline(nullptr, pipelineInfo);
			if (result.result == vk::Result::eSuccess) {
				pipeline = result.value;
				spdlog::info("ComputePass '{}' pipeline created/rebuilt successfully.", name);
			} else {
				spdlog::error("Failed to create compute pipeline for pass '{}'", name);
			}
		};

		if (watcher && computeShader) {
			auto rebuildCb = [this, buildPipeline]() {
				spdlog::info("Rebuilding ComputePass '{}' pipeline due to shader change...", name);
				device.waitIdle();
				buildPipeline();
			};
			watcher->RegisterShader(computeShader, rebuildCb);
		}

		buildPipeline();
	}

	void ComputePass::Dispatch(vk::CommandBuffer cmd, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) const {
		if (pipeline) {
			cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
			cmd.dispatch(groupCountX, groupCountY, groupCountZ);
		}
	}

} // namespace brassica
