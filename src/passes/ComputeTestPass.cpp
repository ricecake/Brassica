#include "passes/ComputeTestPass.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	ComputeTestPass::ComputeTestPass(vk::Device dev, ShaderWatcher* watcher)
		: ComputePass("ComputeTestPass", dev) {
		InitPipeline(dev, watcher);
	}

	void ComputeTestPass::InitPipeline(vk::Device dev, ShaderWatcher* watcher) {
		if (!computeShader.CompileComputeFromFile(dev, "shaders/test.comp")) {
			spdlog::warn("shaders/test.comp not found or failed compilation; initializing ComputeTestPass structure.");
		} else {
			SetShader(&computeShader);
			InitComputePipeline({}, {}, watcher);
		}
	}

	ComputeTestPassData ComputeTestPass::RegisterPass(
		FrameGraph&          fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent3D         texExtent,
		vk::DeviceSize        ssboSize
	) {
		const auto& passData = fg.addCallbackPass<ComputeTestPassData>(
			"ComputeTestPass",
			[&](FrameGraph::Builder& builder, ComputeTestPassData& data) {
				data.outputTexture3D = builder.create<FrameGraphTexture3D>(
					"Volume3D",
					FrameGraphTexture3D::Desc{.extent = texExtent, .format = vk::Format::eR8G8B8A8Unorm}
				);
				data.outputTexture3D = builder.write(data.outputTexture3D, static_cast<uint32_t>(TextureUsage::StorageWrite));

				data.outputSSBO = builder.create<FrameGraphSSBO>(
					"BufferSSBO",
					FrameGraphSSBO::Desc{.size = ssboSize}
				);
				data.outputSSBO = builder.write(data.outputSSBO, static_cast<uint32_t>(BufferUsage::StorageWrite));
			},
			[this, texExtent](const ComputeTestPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				Dispatch(cmd, (texExtent.width + 7) / 8, (texExtent.height + 7) / 8, (texExtent.depth + 7) / 8);
			}
		);

		blackboard.add<ComputeTestPassData>() = passData;
		return passData;
	}

} // namespace brassica
