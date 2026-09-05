#include "passes/DeferredPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"
#include "passes/GradientPass.hpp"

namespace brassica {

	DeferredPass::DeferredPass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher
	) : RenderPass("DeferredPass", dev, colorFmt) {
		CreateDescriptorResources(dev);
		InitPipeline(dev, globalSet0Layout, colorFmt, watcher);
	}

	DeferredPass::~DeferredPass() {
		CleanupDescriptorResources();
	}

	void DeferredPass::CreateDescriptorResources(vk::Device dev) {
		// Set 1 Layout: 4 Samplers (Position, Normal, Albedo, Background)
		std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
		for (uint32_t i = 0; i < 4; ++i) {
			bindings[i].setBinding(i);
			bindings[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
			bindings[i].setDescriptorCount(1);
			bindings[i].setStageFlags(vk::ShaderStageFlagBits::eFragment);
		}

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		gbufferSetLayout = dev.createDescriptorSetLayout(layoutInfo);

		// Pool
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eCombinedImageSampler);
		poolSize.setDescriptorCount(4 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// Allocate Sets
		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, gbufferSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = dev.allocateDescriptorSets(allocInfo);
		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			gbufferDescriptorSets[i] = allocatedSets[i];
		}

		// Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eNearest);
		samplerInfo.setMinFilter(vk::Filter::eNearest);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void DeferredPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (gbufferSetLayout) {
			device.destroyDescriptorSetLayout(gbufferSetLayout);
			gbufferSetLayout = nullptr;
		}
	}

	void DeferredPass::InitPipeline(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher
	) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/deferred.vert")) {
			spdlog::error("Failed to compile deferred.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/deferred.frag")) {
			spdlog::error("Failed to compile deferred.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);

		std::array<vk::DescriptorSetLayout, 2> setLayouts = {globalSet0Layout, gbufferSetLayout};
		InitRenderPipeline(colorFmt, vk::Format::eUndefined, setLayouts, {}, watcher);
	}

	FrameGraphResource DeferredPass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet,
		uint32_t              activeFrame
	) {
		const auto& gbufferData = blackboard.get<GBufferData>();
		const auto& gradientData = blackboard.get<GradientPassData>();
		const auto& swapchainData = blackboard.get<SwapchainData>();

		const auto& passData = fg.addCallbackPass<DeferredPassData>(
			"DeferredPass",
			[&](FrameGraph::Builder& builder, DeferredPassData& data) {
				builder.read(gbufferData.positionTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
				builder.read(gbufferData.normalTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
				builder.read(gbufferData.albedoTarget, static_cast<uint32_t>(TextureUsage::SampledShaderRead));
				builder.read(gradientData.target, static_cast<uint32_t>(TextureUsage::SampledShaderRead));

				data.target = builder.write(swapchainData.target, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet, gbufferData, gradientData, activeFrame](const DeferredPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTex = resources.get<FrameGraphTexture2D>(gbufferData.positionTarget);
				auto& normTex = resources.get<FrameGraphTexture2D>(gbufferData.normalTarget);
				auto& albTex = resources.get<FrameGraphTexture2D>(gbufferData.albedoTarget);
				auto& bgTex = resources.get<FrameGraphTexture2D>(gradientData.target);
				auto& targetTex = resources.get<FrameGraphTexture2D>(data.target);

				vk::DescriptorSet currentGbufferSet = gbufferDescriptorSets[activeFrame % FRAME_OVERLAP];

				std::array<vk::DescriptorImageInfo, 4> imageInfos{};
				imageInfos[0].setSampler(sampler).setImageView(posTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[1].setSampler(sampler).setImageView(normTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[2].setSampler(sampler).setImageView(albTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[3].setSampler(sampler).setImageView(bgTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 4> descriptorWrites{};
				for (uint32_t i = 0; i < 4; ++i) {
					descriptorWrites[i].setDstSet(currentGbufferSet);
					descriptorWrites[i].setDstBinding(i);
					descriptorWrites[i].setDstArrayElement(0);
					descriptorWrites[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
					descriptorWrites[i].setImageInfo(imageInfos[i]);
				}

				device.updateDescriptorSets(descriptorWrites, nullptr);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTex.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}});

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));

				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, globalDescriptorSet, nullptr);
				}
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, currentGbufferSet, nullptr);

				cmd.draw(3, 1, 0, 0);

				EndRendering(cmd);
			}
		);

		blackboard.add<DeferredPassData>() = passData;
		return passData.target;
	}

} // namespace brassica
