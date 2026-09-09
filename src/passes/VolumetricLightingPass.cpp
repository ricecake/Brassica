#include "passes/VolumetricLightingPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"

#include "passes/AtmosphereLUTPass.hpp"
#include "passes/TerrainPass.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	VolumetricLightingPass::VolumetricLightingPass(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	):
		Pass("VolumetricLightingPass", dev) {
		pipelineCache = pCache;
		CreateDescriptorResources(dev);
		InitPipeline(dev, globalSet0Layout, watcher, pCache);
	}

	VolumetricLightingPass::~VolumetricLightingPass() {
		Destroy3DGridResources();
		CleanupDescriptorResources();
		DestroyPipeline();
	}

	void VolumetricLightingPass::Create3DGridResources(VmaAllocator allocator) {
		if (injectionImage || allocator == VK_NULL_HANDLE)
			return;
		lastAllocator = allocator;

		auto create3DTex = [this, allocator](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
			VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			imageInfo.imageType = VK_IMAGE_TYPE_3D;
			imageInfo.extent = VkExtent3D{160, 90, 256}; // 4 cascades * 64
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

			VkImage vkImg = VK_NULL_HANDLE;
			if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &alloc, nullptr) == VK_SUCCESS) {
				img = vkImg;

				vk::ImageViewCreateInfo viewInfo{};
				viewInfo.setImage(img);
				viewInfo.setViewType(vk::ImageViewType::e3D);
				viewInfo.setFormat(vk::Format::eR16G16B16A16Sfloat);
				viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
				view = device.createImageView(viewInfo);
			}
		};

		create3DTex(injectionImage, injectionImageView, injectionAllocation);
		create3DTex(integratedImage, integratedImageView, integratedAllocation);

		// Double-buffered 3D history textures
		create3DTex(history3DImages[0], history3DViews[0], history3DAllocations[0]);
		create3DTex(history3DImages[1], history3DViews[1], history3DAllocations[1]);

		// Create dummy 2D image for unbound transmittance/clipmap
		VkImageCreateInfo img2D{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		img2D.imageType = VK_IMAGE_TYPE_2D;
		img2D.extent = VkExtent3D{1, 1, 1};
		img2D.mipLevels = 1;
		img2D.arrayLayers = 1;
		img2D.format = VK_FORMAT_R8G8B8A8_UNORM;
		img2D.tiling = VK_IMAGE_TILING_OPTIMAL;
		img2D.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		img2D.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
		img2D.samples = VK_SAMPLE_COUNT_1_BIT;

		VmaAllocationCreateInfo alloc2D{};
		alloc2D.usage = VMA_MEMORY_USAGE_AUTO;

		VkImage vk2D = VK_NULL_HANDLE;
		if (vmaCreateImage(allocator, &img2D, &alloc2D, &vk2D, &dummy2DTex.allocation, nullptr) == VK_SUCCESS) {
			dummy2DTex.image = vk2D;
			vk::ImageViewCreateInfo viewInfo{};
			viewInfo.setImage(dummy2DTex.image);
			viewInfo.setViewType(vk::ImageViewType::e2D);
			viewInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
			viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			dummy2DTex.imageView = device.createImageView(viewInfo);
		}

		// Dummy buffer for SSBO binding
		VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufInfo.size = 256;
		bufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

		VkBuffer vkBuf = VK_NULL_HANDLE;
		if (vmaCreateBuffer(allocator, &bufInfo, &alloc2D, &vkBuf, &dummyAllocation, nullptr) == VK_SUCCESS) {
			dummyBuffer = vkBuf;
		}
	}

	void VolumetricLightingPass::Destroy3DGridResources() {
		if (lastAllocator == VK_NULL_HANDLE)
			return;

		auto destroyTex = [this](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
			if (view) {
				device.destroyImageView(view);
				view = nullptr;
			}
			if (img && alloc) {
				vmaDestroyImage(lastAllocator, img, alloc);
				img = nullptr;
				alloc = VK_NULL_HANDLE;
			}
		};

		destroyTex(injectionImage, injectionImageView, injectionAllocation);
		destroyTex(integratedImage, integratedImageView, integratedAllocation);
		destroyTex(history3DImages[0], history3DViews[0], history3DAllocations[0]);
		destroyTex(history3DImages[1], history3DViews[1], history3DAllocations[1]);

		if (dummy2DTex.imageView) {
			device.destroyImageView(dummy2DTex.imageView);
			dummy2DTex.imageView = nullptr;
		}
		if (dummy2DTex.image && dummy2DTex.allocation) {
			vmaDestroyImage(lastAllocator, dummy2DTex.image, dummy2DTex.allocation);
			dummy2DTex.image = nullptr;
			dummy2DTex.allocation = VK_NULL_HANDLE;
		}

		if (dummyBuffer && dummyAllocation) {
			vmaDestroyBuffer(lastAllocator, dummyBuffer, dummyAllocation);
			dummyBuffer = nullptr;
			dummyAllocation = VK_NULL_HANDLE;
		}
	}

	void VolumetricLightingPass::CreateDescriptorResources(vk::Device dev) {
		if (!dev)
			return;

		// Injection Set Layout (Set 1):
		// Binding 0: Storage Image 3D (injectionGrid)
		// Binding 1: Combined Image Sampler 3D (uHistoryTexture)
		// Binding 2: Combined Image Sampler 2D (transmittanceLUT)
		// Binding 3: Combined Image Sampler 2DArray (terrainClipmap)
		// Binding 4: Acceleration Structure (topLevelAS)
		// Binding 5: Storage Buffer (TerrainAABBs)
		std::array<vk::DescriptorSetLayoutBinding, 6> injBindings{};
		injBindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageImage).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		injBindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		injBindings[2].setBinding(2).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		injBindings[3].setBinding(3).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		injBindings[4].setBinding(4).setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		injBindings[5].setBinding(5).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo injLayoutInfo{};
		injLayoutInfo.setBindings(injBindings);
		injectionSetLayout = dev.createDescriptorSetLayout(injLayoutInfo);

		// Integration Set Layout (Set 1):
		// Binding 0: Storage Image 3D (inInjection)
		// Binding 1: Storage Image 3D (outScattering)
		// Binding 2: Storage Image 3D (outHistory)
		std::array<vk::DescriptorSetLayoutBinding, 3> integBindings{};
		integBindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageImage).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		integBindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eStorageImage).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);
		integBindings[2].setBinding(2).setDescriptorType(vk::DescriptorType::eStorageImage).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute);

		vk::DescriptorSetLayoutCreateInfo integLayoutInfo{};
		integLayoutInfo.setBindings(integBindings);
		integrationSetLayout = dev.createDescriptorSetLayout(integLayoutInfo);

		// Descriptor Pool
		std::array<vk::DescriptorPoolSize, 4> poolSizes{};
		poolSizes[0].setType(vk::DescriptorType::eStorageImage).setDescriptorCount(4 * FRAME_OVERLAP);
		poolSizes[1].setType(vk::DescriptorType::eCombinedImageSampler).setDescriptorCount(3 * FRAME_OVERLAP);
		poolSizes[2].setType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1 * FRAME_OVERLAP);
		poolSizes[3].setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1 * FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(2 * FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSizes);
		descriptorPool = dev.createDescriptorPool(poolInfo);

		// Allocate Sets
		std::vector<vk::DescriptorSetLayout> injLayouts(FRAME_OVERLAP, injectionSetLayout);
		vk::DescriptorSetAllocateInfo        injAllocInfo{};
		injAllocInfo.setDescriptorPool(descriptorPool);
		injAllocInfo.setSetLayouts(injLayouts);
		auto allocatedInjSets = dev.allocateDescriptorSets(injAllocInfo);

		std::vector<vk::DescriptorSetLayout> integLayouts(FRAME_OVERLAP, integrationSetLayout);
		vk::DescriptorSetAllocateInfo        integAllocInfo{};
		integAllocInfo.setDescriptorPool(descriptorPool);
		integAllocInfo.setSetLayouts(integLayouts);
		auto allocatedIntegSets = dev.allocateDescriptorSets(integAllocInfo);

		for (size_t i = 0; i < FRAME_OVERLAP; ++i) {
			injectionDescriptorSets[i] = allocatedInjSets[i];
			integrationDescriptorSets[i] = allocatedIntegSets[i];
		}

		// Linear Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		sampler = dev.createSampler(samplerInfo);
	}

	void VolumetricLightingPass::CleanupDescriptorResources() {
		if (sampler) {
			device.destroySampler(sampler);
			sampler = nullptr;
		}
		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (injectionSetLayout) {
			device.destroyDescriptorSetLayout(injectionSetLayout);
			injectionSetLayout = nullptr;
		}
		if (integrationSetLayout) {
			device.destroyDescriptorSetLayout(integrationSetLayout);
			integrationSetLayout = nullptr;
		}
	}

	void VolumetricLightingPass::DestroyPipeline() {
		if (injectionPipeline) {
			device.destroyPipeline(injectionPipeline);
			injectionPipeline = nullptr;
		}
		if (integrationPipeline) {
			device.destroyPipeline(integrationPipeline);
			integrationPipeline = nullptr;
		}
		if (injectionPipelineLayout) {
			device.destroyPipelineLayout(injectionPipelineLayout);
			injectionPipelineLayout = nullptr;
		}
		if (integrationPipelineLayout) {
			device.destroyPipelineLayout(integrationPipelineLayout);
			integrationPipelineLayout = nullptr;
		}
	}

	void VolumetricLightingPass::InitPipeline(
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher,
		vk::PipelineCache       pCache
	) {
		pipelineCache = pCache;

		if (!injectionShader.CompileComputeFromFile(dev, "shaders/volumetric/injection.comp")) {
			spdlog::error("Failed to compile shaders/volumetric/injection.comp");
		}
		if (!integrationShader.CompileComputeFromFile(dev, "shaders/volumetric/integration.comp")) {
			spdlog::error("Failed to compile shaders/volumetric/integration.comp");
		}

		if (!dev)
			return;

		auto buildPipelines = [this, dev, globalSet0Layout]() {
			DestroyPipeline();

			vk::PushConstantRange pushConstantRange{};
			pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eCompute);
			pushConstantRange.setOffset(0);
			pushConstantRange.setSize(sizeof(VolumetricPushConstants));

			// Injection Layout
			std::array<vk::DescriptorSetLayout, 2> injLayouts = {globalSet0Layout, injectionSetLayout};
			vk::PipelineLayoutCreateInfo           injLayoutInfo{};
			injLayoutInfo.setSetLayouts(injLayouts);
			injLayoutInfo.setPushConstantRanges(pushConstantRange);
			injectionPipelineLayout = dev.createPipelineLayout(injLayoutInfo);

			vk::ComputePipelineCreateInfo injPipeInfo{};
			injPipeInfo.setStage(injectionShader.GetStageCreateInfo());
			injPipeInfo.setLayout(injectionPipelineLayout);
			auto injRes = dev.createComputePipeline(pipelineCache, injPipeInfo);
			if (injRes.result == vk::Result::eSuccess) {
				injectionPipeline = injRes.value;
			} else {
				spdlog::error("Failed to create injection compute pipeline");
			}

			// Integration Layout
			std::array<vk::DescriptorSetLayout, 2> integLayouts = {globalSet0Layout, integrationSetLayout};
			vk::PipelineLayoutCreateInfo           integLayoutInfo{};
			integLayoutInfo.setSetLayouts(integLayouts);
			integLayoutInfo.setPushConstantRanges(pushConstantRange);
			integrationPipelineLayout = dev.createPipelineLayout(integLayoutInfo);

			vk::ComputePipelineCreateInfo integPipeInfo{};
			integPipeInfo.setStage(integrationShader.GetStageCreateInfo());
			integPipeInfo.setLayout(integrationPipelineLayout);
			auto integRes = dev.createComputePipeline(pipelineCache, integPipeInfo);
			if (integRes.result == vk::Result::eSuccess) {
				integrationPipeline = integRes.value;
			} else {
				spdlog::error("Failed to create integration compute pipeline");
			}
		};

		if (watcher) {
			auto rebuildCb = [this, buildPipelines]() {
				spdlog::info("Rebuilding VolumetricLightingPass pipelines due to shader modification...");
				device.waitIdle();
				buildPipelines();
			};
			watcher->RegisterShader(&injectionShader, rebuildCb);
			watcher->RegisterShader(&integrationShader, rebuildCb);
		}

		buildPipelines();
	}

	VolumetricLightingData VolumetricLightingPass::RegisterPass(
		FrameGraph&                  fg,
		FrameGraphBlackboard&        blackboard,
		vk::DescriptorSet            globalDescriptorSet,
		uint32_t                     activeFrame,
		vk::ImageView                transmittanceLUTView,
		vk::Sampler                  transmittanceLUTSampler,
		vk::ImageView                clipmapImageView,
		vk::Sampler                  clipmapSampler,
		vk::AccelerationStructureKHR tlas,
		vk::Buffer                   aabbBuffer,
		const VolumetricPushConstants& pushConstants,
		VmaAllocator                 allocator
	) {
		if (allocator != VK_NULL_HANDLE) {
			Create3DGridResources(allocator);
		}

		vk::Extent3D extent{160, 90, 256}; // 4 cascades * 64

		FrameGraphResource importedInjection = fg.import(
			"VolumetricInjectionGrid",
			{extent, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled},
			FrameGraphTexture3D{injectionImage, injectionImageView}
		);

		FrameGraphResource importedIntegrated = fg.import(
			"VolumetricIntegratedGrid",
			{extent, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled},
			FrameGraphTexture3D{integratedImage, integratedImageView}
		);

		struct PassData {
			FrameGraphResource injectionGrid;
			FrameGraphResource integratedGrid;
		};

		const auto& passData = fg.addCallbackPass<PassData>(
			"VolumetricLightingPass",
			[&](FrameGraph::Builder& builder, PassData& data) {
				data.injectionGrid = builder.write(importedInjection, static_cast<uint32_t>(TextureUsage::StorageWrite));
				data.integratedGrid = builder.write(importedIntegrated, static_cast<uint32_t>(TextureUsage::StorageWrite));
				builder.setSideEffect();
			},
			[this,
			 globalDescriptorSet,
			 activeFrame,
			 transmittanceLUTView,
			 transmittanceLUTSampler,
			 clipmapImageView,
			 clipmapSampler,
			 tlas,
			 aabbBuffer,
			 pushConstants](const PassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& injTex = resources.get<FrameGraphTexture3D>(data.injectionGrid);
				auto& integTex = resources.get<FrameGraphTexture3D>(data.integratedGrid);

				vk::DescriptorSet injSet = injectionDescriptorSets[activeFrame % FRAME_OVERLAP];
				vk::DescriptorSet integSet = integrationDescriptorSets[activeFrame % FRAME_OVERLAP];

				uint32_t currentHistIndex = historyIndex3D;
				uint32_t nextHistIndex = 1 - historyIndex3D;

				// Write Injection Descriptors
				vk::DescriptorImageInfo injImageInfo{};
				injImageInfo.setImageView(injTex.imageView).setImageLayout(vk::ImageLayout::eGeneral);

				vk::DescriptorImageInfo historyReadInfo{};
				historyReadInfo.setSampler(sampler)
					.setImageView(history3DViews[currentHistIndex] ? history3DViews[currentHistIndex] : injTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				vk::DescriptorImageInfo transLUTInfo{};
				transLUTInfo.setSampler(transmittanceLUTSampler ? transmittanceLUTSampler : sampler)
					.setImageView(transmittanceLUTView ? transmittanceLUTView : dummy2DTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				vk::DescriptorImageInfo clipmapInfo{};
				clipmapInfo.setSampler(clipmapSampler ? clipmapSampler : sampler)
					.setImageView(clipmapImageView ? clipmapImageView : dummy2DTex.imageView)
					.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

				std::array<vk::WriteDescriptorSet, 6> injWrites{};
				injWrites[0].setDstSet(injSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageImage).setImageInfo(injImageInfo);
				injWrites[1].setDstSet(injSet).setDstBinding(1).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setImageInfo(historyReadInfo);
				injWrites[2].setDstSet(injSet).setDstBinding(2).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setImageInfo(transLUTInfo);
				injWrites[3].setDstSet(injSet).setDstBinding(3).setDescriptorType(vk::DescriptorType::eCombinedImageSampler).setImageInfo(clipmapInfo);

				vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
				if (tlas) {
					asInfo.setAccelerationStructures(tlas);
				}
				injWrites[4].setDstSet(injSet).setDstBinding(4).setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR).setDescriptorCount(1).setPNext(&asInfo);

				vk::DescriptorBufferInfo bufferInfo{};
				bufferInfo.setBuffer(aabbBuffer ? aabbBuffer : dummyBuffer).setOffset(0).setRange(VK_WHOLE_SIZE);
				injWrites[5].setDstSet(injSet).setDstBinding(5).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(bufferInfo);

				device.updateDescriptorSets(injWrites, nullptr);

				// Write Integration Descriptors
				vk::DescriptorImageInfo injReadInfo{};
				injReadInfo.setImageView(injTex.imageView).setImageLayout(vk::ImageLayout::eGeneral);

				vk::DescriptorImageInfo integWriteInfo{};
				integWriteInfo.setImageView(integTex.imageView).setImageLayout(vk::ImageLayout::eGeneral);

				vk::DescriptorImageInfo historyWriteInfo{};
				historyWriteInfo.setImageView(history3DViews[nextHistIndex] ? history3DViews[nextHistIndex] : integTex.imageView)
					.setImageLayout(vk::ImageLayout::eGeneral);

				std::array<vk::WriteDescriptorSet, 3> integWrites{};
				integWrites[0].setDstSet(integSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageImage).setImageInfo(injReadInfo);
				integWrites[1].setDstSet(integSet).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageImage).setImageInfo(integWriteInfo);
				integWrites[2].setDstSet(integSet).setDstBinding(2).setDescriptorType(vk::DescriptorType::eStorageImage).setImageInfo(historyWriteInfo);

				device.updateDescriptorSets(integWrites, nullptr);

				// Dispatch Injection Pass (160x90x256 grid)
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, injectionPipeline);
				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, injectionPipelineLayout, 0, globalDescriptorSet, nullptr);
				}
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, injectionPipelineLayout, 1, injSet, nullptr);

				VolumetricPushConstants activePush = pushConstants;
				if (!hasHistory3D) {
					activePush.params1.y = 0.0f; // Disable temporal blend on first frame
				}
				cmd.pushConstants(injectionPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(VolumetricPushConstants), &activePush);
				cmd.dispatch((160 + 7) / 8, (90 + 7) / 8, (256 + 3) / 4);

				// Memory Barrier between Injection and Integration
				vk::MemoryBarrier memoryBarrier{
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead
				};
				cmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					memoryBarrier,
					nullptr,
					nullptr
				);

				// Dispatch Integration Pass (160x90 threads, 256 local threads along Z)
				cmd.bindPipeline(vk::PipelineBindPoint::eCompute, integrationPipeline);
				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, integrationPipelineLayout, 0, globalDescriptorSet, nullptr);
				}
				cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, integrationPipelineLayout, 1, integSet, nullptr);
				cmd.pushConstants(integrationPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(VolumetricPushConstants), &activePush);
				cmd.dispatch(160, 90, 1);

				// Advance 3D history index
				historyIndex3D = nextHistIndex;
				hasHistory3D = true;
			}
		);

		VolumetricLightingData resData{
			.injectionGrid = passData.injectionGrid,
			.integratedGrid = passData.integratedGrid
		};
		blackboard.add<VolumetricLightingData>() = resData;
		return resData;
	}

} // namespace brassica
