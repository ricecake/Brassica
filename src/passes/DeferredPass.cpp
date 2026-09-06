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
		// Set 1 Layout:
		// Binding 0: Position sampler
		// Binding 1: Normal sampler
		// Binding 2: Albedo sampler
		// Binding 3: Background sampler
		// Binding 4: Terrain Clipmap Texture Array sampler
		// Binding 5: Acceleration Structure (TLAS)
		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
		for (uint32_t i = 0; i < 5; ++i) {
			bindings[i].setBinding(i);
			bindings[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
			bindings[i].setDescriptorCount(1);
			bindings[i].setStageFlags(vk::ShaderStageFlagBits::eFragment);
		}
		bindings[5].setBinding(5);
		bindings[5].setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
		bindings[5].setDescriptorCount(1);
		bindings[5].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		gbufferSetLayout = dev.createDescriptorSetLayout(layoutInfo);

		// Pool
		std::array<vk::DescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(5 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
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
		InitRenderPipeline(colorFmt, vk::Format::eUndefined, setLayouts, {}, watcher, false, false, vk::CompareOp::eLess, vk::CullModeFlagBits::eNone);
	}

	FrameGraphResource DeferredPass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet,
		uint32_t              activeFrame,
		vk::ImageView         clipmapImageView,
		vk::Sampler           clipmapSampler,
		vk::AccelerationStructureKHR tlas
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
			[this, extent, globalDescriptorSet, gbufferData, gradientData, activeFrame, clipmapImageView, clipmapSampler, tlas](const DeferredPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTex = resources.get<FrameGraphTexture2D>(gbufferData.positionTarget);
				auto& normTex = resources.get<FrameGraphTexture2D>(gbufferData.normalTarget);
				auto& albTex = resources.get<FrameGraphTexture2D>(gbufferData.albedoTarget);
				auto& bgTex = resources.get<FrameGraphTexture2D>(gradientData.target);
				auto& targetTex = resources.get<FrameGraphTexture2D>(data.target);

				vk::DescriptorSet currentGbufferSet = gbufferDescriptorSets[activeFrame % FRAME_OVERLAP];

				std::array<vk::DescriptorImageInfo, 5> imageInfos{};
				imageInfos[0].setSampler(sampler).setImageView(posTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[1].setSampler(sampler).setImageView(normTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[2].setSampler(sampler).setImageView(albTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[3].setSampler(sampler).setImageView(bgTex.imageView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				vk::Sampler clipSampler = clipmapSampler ? clipmapSampler : sampler;
				vk::ImageView clipView = clipmapImageView ? clipmapImageView : posTex.imageView;
				imageInfos[4].setSampler(clipSampler).setImageView(clipView).setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 6> descriptorWrites{};
				for (uint32_t i = 0; i < 5; ++i) {
					descriptorWrites[i].setDstSet(currentGbufferSet);
					descriptorWrites[i].setDstBinding(i);
					descriptorWrites[i].setDstArrayElement(0);
					descriptorWrites[i].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
					descriptorWrites[i].setImageInfo(imageInfos[i]);
				}

				vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
				if (tlas) {
					asInfo.setAccelerationStructures(tlas);
				}

				descriptorWrites[5].setDstSet(currentGbufferSet);
				descriptorWrites[5].setDstBinding(5);
				descriptorWrites[5].setDstArrayElement(0);
				descriptorWrites[5].setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
				descriptorWrites[5].setDescriptorCount(1);
				descriptorWrites[5].setPNext(&asInfo);

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
