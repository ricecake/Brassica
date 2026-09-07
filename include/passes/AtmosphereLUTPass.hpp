#pragma once

#include <array>
#include <vector>

#include "vulkan/vulkan.hpp"

#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/ComputePass.hpp"
#include "passes/PassResource.hpp"
#include "Shader.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	struct AtmosphereLUTData {
		FrameGraphResource transmittanceLUT;
		FrameGraphResource multiScatteringLUT;
	};

	class AtmosphereLUTPass : public Pass {
	public:
		AtmosphereLUTPass(vk::Device device, ShaderWatcher* watcher = nullptr);
		~AtmosphereLUTPass() override;

		void InitPipeline(vk::Device device, ShaderWatcher* watcher = nullptr);

		AtmosphereLUTData RegisterPass(
			FrameGraph&                     fg,
			FrameGraphBlackboard&           blackboard,
			uint32_t                        activeFrame,
			const AtmospherePushConstants& push = {}
		);

		void DestroyPipeline();

	private:
		static constexpr uint32_t FRAME_OVERLAP = 2;

		ComputeShader transmittanceShader;
		ComputeShader multiScatteringShader;

		vk::DescriptorSetLayout transmittanceSetLayout{nullptr};
		vk::DescriptorSetLayout multiScatteringSetLayout{nullptr};
		vk::DescriptorPool      descriptorPool{nullptr};

		vk::DescriptorSet transmittanceSets[FRAME_OVERLAP]{nullptr, nullptr};
		vk::DescriptorSet multiScatteringSets[FRAME_OVERLAP]{nullptr, nullptr};

		vk::PipelineLayout transmittancePipelineLayout{nullptr};
		vk::PipelineLayout multiScatteringPipelineLayout{nullptr};

		vk::Pipeline transmittancePipeline{nullptr};
		vk::Pipeline multiScatteringPipeline{nullptr};

		vk::Sampler sampler{nullptr};

		void CreateDescriptorResources(vk::Device dev);
		void CleanupDescriptorResources();
	};

} // namespace brassica
