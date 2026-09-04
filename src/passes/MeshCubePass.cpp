#include "passes/MeshCubePass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	MeshCubePass::MeshCubePass(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) : RenderPass(
			"MeshCubePass",
			dev,
			std::array<vk::Format, 3>{vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm},
			vk::Format::eD32Sfloat
		) {
		InitPipeline(instance, dev, globalSet0Layout, watcher);
	}

	void MeshCubePass::InitPipeline(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
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

		std::array<vk::Format, 3> colorFmts = {
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR8G8B8A8Unorm
		};

		InitRenderPipeline(
			colorFmts,
			vk::Format::eD32Sfloat,
			std::span(&globalSet0Layout, 1),
			{},
			watcher,
			true,  // depthTestEnable
			true,  // depthWriteEnable
			vk::CompareOp::eLess
		);
	}

	void MeshCubePass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet
	) {
		const auto& passData = fg.addCallbackPass<MeshCubePassData>(
			"MeshCubePass",
			[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
				data.positionTarget = builder.create<FrameGraphTexture2D>(
					"GBuffer_Position",
					FrameGraphTexture2D::Desc{
						.extent = extent,
						.format = vk::Format::eR16G16B16A16Sfloat,
						.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
					}
				);
				data.positionTarget = builder.write(data.positionTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

				data.normalTarget = builder.create<FrameGraphTexture2D>(
					"GBuffer_Normal",
					FrameGraphTexture2D::Desc{
						.extent = extent,
						.format = vk::Format::eR16G16B16A16Sfloat,
						.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
					}
				);
				data.normalTarget = builder.write(data.normalTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

				data.albedoTarget = builder.create<FrameGraphTexture2D>(
					"GBuffer_Albedo",
					FrameGraphTexture2D::Desc{
						.extent = extent,
						.format = vk::Format::eR8G8B8A8Unorm,
						.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
					}
				);
				data.albedoTarget = builder.write(data.albedoTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));

				data.depthTarget = builder.create<FrameGraphTexture2D>(
					"GBuffer_Depth",
					FrameGraphTexture2D::Desc{
						.extent = extent,
						.format = vk::Format::eD32Sfloat,
						.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
					}
				);
				data.depthTarget = builder.write(data.depthTarget, static_cast<uint32_t>(TextureUsage::DepthStencilAttachment));

				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet](const MeshCubePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTex = resources.get<FrameGraphTexture2D>(data.positionTarget);
				auto& normTex = resources.get<FrameGraphTexture2D>(data.normalTarget);
				auto& albTex = resources.get<FrameGraphTexture2D>(data.albedoTarget);
				auto& depthTex = resources.get<FrameGraphTexture2D>(data.depthTarget);

				std::array<vk::RenderingAttachmentInfo, 3> colorAttachments{};

				// Clear position, normal, albedo to zeros
				for (int i = 0; i < 3; ++i) {
					colorAttachments[i].setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
					colorAttachments[i].setLoadOp(vk::AttachmentLoadOp::eClear);
					colorAttachments[i].setStoreOp(vk::AttachmentStoreOp::eStore);
					colorAttachments[i].setClearValue(vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}});
				}

				colorAttachments[0].setImageView(posTex.imageView);
				colorAttachments[1].setImageView(normTex.imageView);
				colorAttachments[2].setImageView(albTex.imageView);

				vk::RenderingAttachmentInfo depthAttachmentInfo{};
				depthAttachmentInfo.setImageView(depthTex.imageView);
				depthAttachmentInfo.setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
				depthAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
				depthAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
				depthAttachmentInfo.setClearValue(vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}});

				BeginRendering(cmd, extent, colorAttachments, &depthAttachmentInfo);

				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, globalDescriptorSet, nullptr);
				}

				cmd.drawMeshTasksEXT(1, 1, 1, dls);

				EndRendering(cmd);
			}
		);

		blackboard.add<MeshCubePassData>() = passData;
		blackboard.add<GBufferData>() = GBufferData{
			.positionTarget = passData.positionTarget,
			.normalTarget = passData.normalTarget,
			.albedoTarget = passData.albedoTarget,
			.depthTarget = passData.depthTarget
		};
	}

} // namespace brassica
