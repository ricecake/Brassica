#include "passes/GradientPass.hpp"

#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	GradientPass::GradientPass(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher)
		: RenderPass("GradientPass", dev, colorFmt) {
		InitPipeline(dev, colorFmt, watcher);
	}

	void GradientPass::InitPipeline(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/gradient.vert")) {
			spdlog::error("Failed to compile gradient.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/gradient.frag")) {
			spdlog::error("Failed to compile gradient.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);
		InitRenderPipeline(colorFmt, vk::Format::eUndefined, {}, {}, watcher);
	}

	FrameGraphResource GradientPass::RegisterPass(FrameGraph& fg, FrameGraphBlackboard& blackboard, vk::Extent2D extent) {
		const auto& swapchainData = blackboard.get<SwapchainData>();
		const auto& passData = fg.addCallbackPass<GradientPassData>(
			"GradientPass",
			[&](FrameGraph::Builder& builder, GradientPassData& data) {
				data.target = builder.write(
					swapchainData.target,
					static_cast<uint32_t>(TextureUsage::ColorAttachment)
				);
				builder.setSideEffect();
			},
			[this, extent](const GradientPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             targetTexture = resources.get<FrameGraphTexture2D>(data.target);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTexture.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(
					vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}}
				);

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));
				cmd.draw(3, 1, 0, 0);
				EndRendering(cmd);
			}
		);
		blackboard.add<GradientPassData>() = passData;
		return passData.target;
	}

} // namespace brassica
