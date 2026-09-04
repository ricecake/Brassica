#pragma once

#include "vulkan/vulkan.hpp"
#include "fg/Blackboard.hpp"
#include "fg/FrameGraph.hpp"
#include "passes/ComputePass.hpp"
#include "passes/PassResource.hpp"

namespace brassica {

	class ShaderWatcher;

	struct ComputeTestPassData {
		FrameGraphResource outputTexture3D;
		FrameGraphResource outputSSBO;
	};

	class ComputeTestPass : public ComputePass {
	public:
		ComputeTestPass(vk::Device device, ShaderWatcher* watcher = nullptr);
		~ComputeTestPass() override = default;

		void InitPipeline(vk::Device device, ShaderWatcher* watcher = nullptr);

		ComputeTestPassData RegisterPass(FrameGraph& fg, FrameGraphBlackboard& blackboard, vk::Extent3D texExtent, vk::DeviceSize ssboSize);

	private:
		ComputeShader computeShader;
	};

} // namespace brassica
