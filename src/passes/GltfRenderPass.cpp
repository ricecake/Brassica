#include "passes/GltfRenderPass.hpp"
#include "spdlog/spdlog.h"
#include <array>

namespace brassica {

	GltfRenderPass::GltfRenderPass(
		vk::Instance instance,
		vk::Device dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::DescriptorSetLayout modelGeometrySetLayout,
		vk::DescriptorSetLayout textureSetLayout,
		ShaderWatcher* watcher
	) : RenderPass(
			"GltfRenderPass",
			dev,
			std::array<vk::Format, 3>{vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm},
			vk::Format::eD32Sfloat
		) {
		InitPipeline(instance, dev, globalSet0Layout, modelGeometrySetLayout, textureSetLayout, watcher);
	}

	void GltfRenderPass::InitPipeline(
		vk::Instance instance,
		vk::Device dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::DescriptorSetLayout modelGeometrySetLayout,
		vk::DescriptorSetLayout textureSetLayout,
		ShaderWatcher* watcher
	) {
		dls.init(instance, dev);

		if (!meshShader.CompileMeshFromFile(dev, "shaders/gltf_model.mesh")) {
			spdlog::error("Failed to compile shaders/gltf_model.mesh");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/gltf_model.frag")) {
			spdlog::error("Failed to compile shaders/gltf_model.frag");
		}

		SetShaders(&meshShader, &fragShader);

		modelSetLayout = modelGeometrySetLayout;
		if (!modelSetLayout) {
			std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
			for (uint32_t i = 0; i < 8; ++i) {
				bindings[i].setBinding(i)
					.setDescriptorType(vk::DescriptorType::eStorageBuffer)
					.setDescriptorCount(1)
					.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment);
			}
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(bindings);
			modelSetLayout = device.createDescriptorSetLayout(layoutInfo);
		}

		std::array<vk::DescriptorSetLayout, 3> setLayouts = {
			globalSet0Layout,
			modelSetLayout,
			textureSetLayout
		};

		vk::PushConstantRange pcRange{};
		pcRange.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT);
		pcRange.setOffset(0);
		pcRange.setSize(sizeof(GltfRenderPushConstants));

		std::array<vk::Format, 3> colorFmts = {
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR8G8B8A8Unorm
		};

		InitRenderPipeline(
			colorFmts,
			vk::Format::eD32Sfloat,
			setLayouts,
			std::span(&pcRange, 1),
			watcher,
			true,  // enableDepthTest
			true,  // enableDepthWrite
			vk::CompareOp::eLess
		);
	}

	void GltfRenderPass::RegisterPass(
		FrameGraph& fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D extent,
		vk::DescriptorSet globalDescriptorSet,
		vk::DescriptorSet modelGeometrySet,
		vk::DescriptorSet textureSet,
		vk::Buffer indirectDrawBuffer,
		vk::Buffer drawCountBuffer,
		uint32_t maxDrawCount,
		const GltfRenderPushConstants& pushConstants
	) {
		if (!blackboard.has<GBufferData>()) return;
		const auto& gbuffer = blackboard.get<GBufferData>();

		fg.addCallbackPass<GBufferData>(
			"GltfRenderPass",
			[&](FrameGraph::Builder& builder, GBufferData& data) {
				(void)builder.write(gbuffer.positionTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				(void)builder.write(gbuffer.normalTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				(void)builder.write(gbuffer.albedoTarget, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				(void)builder.write(gbuffer.depthTarget, static_cast<uint32_t>(TextureUsage::DepthStencilAttachment));
				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet, modelGeometrySet, textureSet, indirectDrawBuffer, drawCountBuffer, maxDrawCount, pushConstants](
				const GBufferData& data,
				FrameGraphPassResources& resources,
				void* ctx
			) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				if (maxDrawCount == 0 || !indirectDrawBuffer) return;

				auto& posTex = resources.get<FrameGraphTexture2D>(data.positionTarget);
				auto& normTex = resources.get<FrameGraphTexture2D>(data.normalTarget);
				auto& albTex = resources.get<FrameGraphTexture2D>(data.albedoTarget);
				auto& depthTex = resources.get<FrameGraphTexture2D>(data.depthTarget);

				std::array<vk::RenderingAttachmentInfo, 3> colorAttachments{};
				colorAttachments[0].setImageView(posTex.imageView).setImageLayout(vk::ImageLayout::eColorAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eLoad).setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachments[1].setImageView(normTex.imageView).setImageLayout(vk::ImageLayout::eColorAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eLoad).setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachments[2].setImageView(albTex.imageView).setImageLayout(vk::ImageLayout::eColorAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eLoad).setStoreOp(vk::AttachmentStoreOp::eStore);

				vk::RenderingAttachmentInfo depthAttachmentInfo{};
				depthAttachmentInfo.setImageView(depthTex.imageView).setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal).setLoadOp(vk::AttachmentLoadOp::eLoad).setStoreOp(vk::AttachmentStoreOp::eStore);

				BeginRendering(cmd, extent, colorAttachments, &depthAttachmentInfo);

				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

				std::array<vk::DescriptorSet, 3> sets = {globalDescriptorSet, modelGeometrySet, textureSet};
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, sets, nullptr);

				cmd.pushConstants<GltfRenderPushConstants>(pipelineLayout, vk::ShaderStageFlagBits::eMeshEXT, 0, pushConstants);

				if (drawCountBuffer) {
					cmd.drawMeshTasksIndirectCountEXT(indirectDrawBuffer, 0, drawCountBuffer, 0, maxDrawCount, sizeof(VkDrawMeshTasksIndirectCommandEXT), dls);
				} else {
					DrawMeshTasksIndirectEXT(cmd, indirectDrawBuffer, 0, maxDrawCount, sizeof(VkDrawMeshTasksIndirectCommandEXT), dls);
				}

				EndRendering(cmd);
			}
		);
	}

} // namespace brassica
