#pragma once

#include "passes/ComputePass.hpp"
#include "Shader.hpp"
#include "fg/FrameGraph.hpp"
#include "fg/Blackboard.hpp"
#include <glm/glm.hpp>

namespace brassica {

	struct GltfComputePushConstants {
		glm::mat4 viewProj{1.0f};
		glm::vec4 frustumPlanes[6];
		uint32_t totalInstances{0};
		uint32_t _pad0{0};
		uint32_t _pad1{0};
		uint32_t _pad2{0};
	};

	struct GltfComputePassData {
		FrameGraphResource indirectDrawTarget;
		FrameGraphResource visibleInstancesTarget;
		FrameGraphResource drawCountTarget;
	};

	class GltfComputePass : public ComputePass {
	public:
		GltfComputePass(vk::Device device, ShaderWatcher* watcher = nullptr);
		~GltfComputePass() override = default;

		void InitPipeline(ShaderWatcher* watcher = nullptr);

		void RegisterPass(
			FrameGraph& fg,
			FrameGraphBlackboard& blackboard,
			vk::DescriptorSet computeSet,
			const GltfComputePushConstants& pushConstants,
			vk::Buffer indirectDrawBuffer,
			vk::Buffer visibleInstanceBuffer,
			vk::Buffer drawCountBuffer,
			size_t indirectDrawSize,
			size_t visibleInstanceSize
		);

		vk::DescriptorSetLayout GetDescriptorSetLayout() const { return computeSetLayout; }

	private:
		ComputeShader computeShader;
		vk::DescriptorSetLayout computeSetLayout{nullptr};
	};

} // namespace brassica
