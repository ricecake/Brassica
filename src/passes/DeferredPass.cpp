#include "passes/DeferredPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"

#include "passes/GradientPass.hpp"
#include "passes/TerrainPass.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	DeferredPass::DeferredPass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFmt,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	):
		RenderPass("DeferredPass", dev, colorFmt) {
		CreateDescriptorResources(dev);
		InitPipeline(dev, globalSet0Layout, colorFmt, watcher, pCache);
	}

	DeferredPass::~DeferredPass() {
		CleanupDescriptorResources();
	}

	void DeferredPass::EnsureDummyBuffer(VmaAllocator allocator) {
		if (dummyBuffer || allocator == VK_NULL_HANDLE)
			return;
		lastAllocator = allocator;

		VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufInfo.size = 256;
		bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		VkBuffer vkBuf = VK_NULL_HANDLE;
		if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &vkBuf, &dummyAllocation, nullptr) == VK_SUCCESS) {
			dummyBuffer = vkBuf;
		}

		if (!dummy3DImage) {
			VkImageCreateInfo img3D{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			img3D.imageType = VK_IMAGE_TYPE_3D;
			img3D.extent = VkExtent3D{1, 1, 1};
			img3D.mipLevels = 1;
			img3D.arrayLayers = 1;
			img3D.format = VK_FORMAT_R16G16B16A16_SFLOAT;
			img3D.tiling = VK_IMAGE_TILING_OPTIMAL;
			img3D.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			img3D.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
			img3D.samples = VK_SAMPLE_COUNT_1_BIT;

			VkImage vk3D = VK_NULL_HANDLE;
			if (vmaCreateImage(allocator, &img3D, &allocInfo, &vk3D, &dummy3DAllocation, nullptr) == VK_SUCCESS) {
				dummy3DImage = vk3D;
				vk::ImageViewCreateInfo viewInfo{};
				viewInfo.setImage(dummy3DImage);
				viewInfo.setViewType(vk::ImageViewType::e3D);
				viewInfo.setFormat(vk::Format::eR16G16B16A16Sfloat);
				viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
				dummy3DView = device.createImageView(viewInfo);
			}
		}
	}

	void DeferredPass::CreateDescriptorResources(vk::Device dev) {
		// Set 1 Layout:
		// Binding 0: Position sampler
		// Binding 1: Normal sampler
		// Binding 2: Albedo sampler
		// Binding 3: Background sampler
		// Binding 4: Terrain Clipmap Texture Array sampler
		// Binding 5: Acceleration Structure (TLAS)
		// Binding 6: Terrain AABBs SSBO
		// Binding 7: Volumetric Integrated Grid 3D sampler
		std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
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

		bindings[6].setBinding(6);
		bindings[6].setDescriptorType(vk::DescriptorType::eStorageBuffer);
		bindings[6].setDescriptorCount(1);
		bindings[6].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		bindings[7].setBinding(7);
		bindings[7].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		bindings[7].setDescriptorCount(1);
		bindings[7].setStageFlags(vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(bindings);
		gbufferSetLayout = dev.createDescriptorSetLayout(layoutInfo);

		// Pool
		std::array<vk::DescriptorPoolSize, 3> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(6 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1 * FRAME_OVERLAP);
		poolSizes[2].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// Allocate Sets
		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, gbufferSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{};
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
		if (dummyBuffer && lastAllocator) {
			vmaDestroyBuffer(lastAllocator, dummyBuffer, dummyAllocation);
			dummyBuffer = nullptr;
			dummyAllocation = VK_NULL_HANDLE;
		}
		if (dummy3DView) {
			device.destroyImageView(dummy3DView);
			dummy3DView = nullptr;
		}
		if (dummy3DImage && lastAllocator) {
			vmaDestroyImage(lastAllocator, dummy3DImage, dummy3DAllocation);
			dummy3DImage = nullptr;
			dummy3DAllocation = VK_NULL_HANDLE;
		}
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
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/deferred.vert")) {
			spdlog::error("Failed to compile deferred.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/deferred.frag")) {
			spdlog::error("Failed to compile deferred.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);

		std::array<vk::DescriptorSetLayout, 2> setLayouts = {globalSet0Layout, gbufferSetLayout};
		vk::PushConstantRange                  pushConstantRange{};
		pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eFragment);
		pushConstantRange.setOffset(0);
		pushConstantRange.setSize(sizeof(TerrainPushConstants));

		storedPushConstants.assign({pushConstantRange});

		InitRenderPipeline(
			colorFmt,
			vk::Format::eUndefined,
			setLayouts,
			storedPushConstants,
			watcher,
			false,
			false,
			vk::CompareOp::eLess,
			vk::CullModeFlagBits::eNone,
			pCache
		);
	}

	FrameGraphResource DeferredPass::RegisterPass(
		FrameGraph&                  fg,
		FrameGraphBlackboard&        blackboard,
		vk::Extent2D                 extent,
		vk::DescriptorSet            globalDescriptorSet,
		uint32_t                     activeFrame,
		vk::ImageView                clipmapImageView,
		vk::Sampler                  clipmapSampler,
		vk::AccelerationStructureKHR tlas,
		vk::Buffer                   aabbBuffer,
		vk::ImageView                volumetricIntegratedView,
		const TerrainPushConstants&  pushConstants,
		VmaAllocator                 allocator
	) {
		if (allocator != VK_NULL_HANDLE) {
			EnsureDummyBuffer(allocator);
		}
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
			[this,
			 extent,
			 globalDescriptorSet,
			 gbufferData,
			 gradientData,
			 activeFrame,
			 clipmapImageView,
			 clipmapSampler,
			 tlas,
			 aabbBuffer,
			 volumetricIntegratedView,
			 pushConstants](const DeferredPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTex = resources.get<FrameGraphTexture2D>(gbufferData.positionTarget);
				auto& normTex = resources.get<FrameGraphTexture2D>(gbufferData.normalTarget);
				auto& albTex = resources.get<FrameGraphTexture2D>(gbufferData.albedoTarget);
				auto& bgTex = resources.get<FrameGraphTexture2D>(gradientData.target);
				auto& targetTex = resources.get<FrameGraphTexture2D>(data.target);

				vk::DescriptorSet currentGbufferSet = gbufferDescriptorSets[activeFrame % FRAME_OVERLAP];

				std::array<vk::DescriptorImageInfo, 6> imageInfos{};
				imageInfos[0]
					.setSampler(sampler)
					.setImageView(posTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[1]
					.setSampler(sampler)
					.setImageView(normTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[2]
					.setSampler(sampler)
					.setImageView(albTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
				imageInfos[3]
					.setSampler(sampler)
					.setImageView(bgTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				vk::Sampler   clipSampler = clipmapSampler ? clipmapSampler : sampler;
				vk::ImageView clipView = clipmapImageView ? clipmapImageView : posTex.imageView;
				imageInfos[4]
					.setSampler(clipSampler)
					.setImageView(clipView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				imageInfos[5]
					.setSampler(sampler)
					.setImageView(volumetricIntegratedView ? volumetricIntegratedView : dummy3DView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 8> descriptorWrites{};
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

				vk::DescriptorBufferInfo bufferInfo{};
				bufferInfo.setBuffer(aabbBuffer ? aabbBuffer : dummyBuffer);
				bufferInfo.setOffset(0);
				bufferInfo.setRange(VK_WHOLE_SIZE);

				descriptorWrites[6].setDstSet(currentGbufferSet);
				descriptorWrites[6].setDstBinding(6);
				descriptorWrites[6].setDstArrayElement(0);
				descriptorWrites[6].setDescriptorType(vk::DescriptorType::eStorageBuffer);
				descriptorWrites[6].setDescriptorCount(1);
				descriptorWrites[6].setBufferInfo(bufferInfo);

				descriptorWrites[7].setDstSet(currentGbufferSet);
				descriptorWrites[7].setDstBinding(7);
				descriptorWrites[7].setDstArrayElement(0);
				descriptorWrites[7].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
				descriptorWrites[7].setImageInfo(imageInfos[5]);

				device.updateDescriptorSets(descriptorWrites, nullptr);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTex.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(
					vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}}
				);

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));

				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(
						vk::PipelineBindPoint::eGraphics,
						pipelineLayout,
						0,
						globalDescriptorSet,
						nullptr
					);
				}
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, currentGbufferSet, nullptr);

				cmd.pushConstants(
					pipelineLayout,
					vk::ShaderStageFlagBits::eFragment,
					0,
					sizeof(TerrainPushConstants),
					&pushConstants
				);

				cmd.draw(3, 1, 0, 0);

				EndRendering(cmd);
			}
		);

		blackboard.add<DeferredPassData>() = passData;
		return passData.target;
	}

} // namespace brassica
