#include "passes/MeshCubePass.hpp"

#include <vector>

#include "spdlog/spdlog.h"

#include "ShaderWatcher.hpp"
#include "passes/GradientPass.hpp"

namespace brassica {

	MeshCubePass::MeshCubePass(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher
	) : RenderPass("MeshCubePass", dev, colorFmt) {
		InitPipeline(instance, dev, globalSet0Layout, colorFmt, watcher);
	}

	void MeshCubePass::InitPipeline(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher
	) {
		dls.init(instance, dev);

		if (!meshShader.CompileMeshFromFile(dev, "shaders/cube.mesh")) {
			spdlog::error("Failed to compile cube.mesh shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/cube.frag")) {
			spdlog::error("Failed to compile cube.frag shader file");
		}

		SetShaders(&meshShader, &fragShader);
		InitRenderPipeline(colorFmt, vk::Format::eUndefined, std::span(&globalSet0Layout, 1), {}, watcher);
	}

	FrameGraphResource MeshCubePass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet
	) {
		FrameGraphResource inputResource;
		if (const auto* gradientData = blackboard.try_get<GradientPassData>()) {
			inputResource = gradientData->target;
		} else {
			inputResource = blackboard.get<SwapchainData>().target;
		}

		const auto& passData = fg.addCallbackPass<MeshCubePassData>(
			"MeshCubePass",
			[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
				data.target = builder.write(
					inputResource,
					static_cast<uint32_t>(TextureUsage::ColorAttachment)
				);
				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet](const MeshCubePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             targetTexture = resources.get<FrameGraphTexture2D>(data.target);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTexture.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eLoad);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));

				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, globalDescriptorSet, nullptr);
				}

				cmd.drawMeshTasksEXT(1, 1, 1, dls);

				EndRendering(cmd);
			}
		);
		blackboard.add<MeshCubePassData>() = passData;
		return passData.target;
	}

} // namespace brassica
