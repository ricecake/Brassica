#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "gltf/GltfTextureManager.hpp"
#include <filesystem>
#include "spdlog/spdlog.h"

namespace brassica {

	GltfTextureManager::~GltfTextureManager() {
		Cleanup();
	}

	void GltfTextureManager::Init(vk::Device dev, VmaAllocator alloc, vk::Queue queue, uint32_t queueFamily) {
		Cleanup();

		device = dev;
		allocator = alloc;
		graphicsQueue = queue;
		queueFamilyIndex = queueFamily;

		// 1. Create Sampler
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
		samplerInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
		sampler = device.createSampler(samplerInfo);

		// 2. Create Descriptor Set Layout
		vk::DescriptorSetLayoutBinding binding{};
		binding.setBinding(0);
		binding.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		binding.setDescriptorCount(MAX_TEXTURES);
		binding.setStageFlags(vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorBindingFlags bindingFlags = vk::DescriptorBindingFlagBits::ePartiallyBound;
		vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{};
		flagsCreateInfo.setBindingFlags(bindingFlags);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(binding);
		layoutInfo.setPNext(&flagsCreateInfo);

		descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);

		// 3. Create Descriptor Pool & Allocate Set
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eCombinedImageSampler);
		poolSize.setDescriptorCount(MAX_TEXTURES);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(1);
		poolInfo.setPoolSizes(poolSize);
		poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind);

		descriptorPool = device.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(descriptorSetLayout);

		descriptorSet = device.allocateDescriptorSets(allocInfo).front();

		// 4. Create Default White Texture at Index 0
		CreateDefaultTexture();
	}

	void GltfTextureManager::Cleanup() {
		if (device) {
			for (auto& tex : textures) {
				if (tex.imageView) device.destroyImageView(tex.imageView);
				if (tex.image && tex.allocation) vmaDestroyImage(allocator, tex.image, tex.allocation);
			}
			textures.clear();

			if (descriptorPool) device.destroyDescriptorPool(descriptorPool);
			if (descriptorSetLayout) device.destroyDescriptorSetLayout(descriptorSetLayout);
			if (sampler) device.destroySampler(sampler);
		}

		descriptorPool = nullptr;
		descriptorSetLayout = nullptr;
		descriptorSet = nullptr;
		sampler = nullptr;
		device = nullptr;
		allocator = VK_NULL_HANDLE;
	}

	void GltfTextureManager::CreateDefaultTexture() {
		uint8_t whitePixel[4] = {255, 255, 255, 255};
		CreateTextureFromPixels(whitePixel, 1, 1, 4, vk::Format::eR8G8B8A8Unorm);
	}

	int32_t GltfTextureManager::CreateTextureFromPixels(
		const uint8_t* pixels,
		int width,
		int height,
		int channels,
		vk::Format format
	) {
		if (!pixels || width <= 0 || height <= 0 || textures.size() >= MAX_TEXTURES) {
			return 0; // fallback to default texture
		}

		VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

		// 1. Create Staging Buffer
		VkBufferCreateInfo stagingBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		stagingBufferInfo.size = imageSize;
		stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

		VmaAllocationCreateInfo stagingAllocInfo{};
		stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer stagingBuffer = VK_NULL_HANDLE;
		VmaAllocation stagingAlloc = VK_NULL_HANDLE;
		VmaAllocationInfo stagingResult{};

		if (vmaCreateBuffer(allocator, &stagingBufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAlloc, &stagingResult) != VK_SUCCESS) {
			return 0;
		}

		if (channels == 4) {
			std::memcpy(stagingResult.pMappedData, pixels, imageSize);
		} else if (channels == 3) {
			auto* dst = static_cast<uint8_t*>(stagingResult.pMappedData);
			for (int i = 0; i < width * height; ++i) {
				dst[i * 4 + 0] = pixels[i * 3 + 0];
				dst[i * 4 + 1] = pixels[i * 3 + 1];
				dst[i * 4 + 2] = pixels[i * 3 + 2];
				dst[i * 4 + 3] = 255;
			}
		}

		// 2. Create GPU Image
		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = VkExtent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = static_cast<VkFormat>(format);
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo imgAllocInfo{};
		imgAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		LoadedTextureResource texRes{};
		VkImage rawImg = VK_NULL_HANDLE;

		if (vmaCreateImage(allocator, &imageInfo, &imgAllocInfo, &rawImg, &texRes.allocation, nullptr) != VK_SUCCESS) {
			vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);
			return 0;
		}
		texRes.image = rawImg;

		// 3. Copy staging buffer to GPU image
		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.setQueueFamilyIndex(queueFamilyIndex);
		poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eTransient);
		vk::CommandPool cmdPool = device.createCommandPool(poolInfo);

		vk::CommandBufferAllocateInfo cmdAlloc{};
		cmdAlloc.setCommandPool(cmdPool);
		cmdAlloc.setLevel(vk::CommandBufferLevel::ePrimary);
		cmdAlloc.setCommandBufferCount(1);
		vk::CommandBuffer cmd = device.allocateCommandBuffers(cmdAlloc).front();

		cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		vk::ImageMemoryBarrier barrierToCopy{};
		barrierToCopy.setOldLayout(vk::ImageLayout::eUndefined);
		barrierToCopy.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
		barrierToCopy.setImage(texRes.image);
		barrierToCopy.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		barrierToCopy.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);

		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			{}, nullptr, nullptr, barrierToCopy
		);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1));
		copyRegion.setImageExtent(vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1});

		cmd.copyBufferToImage(stagingBuffer, texRes.image, vk::ImageLayout::eTransferDstOptimal, copyRegion);

		vk::ImageMemoryBarrier barrierToShader{};
		barrierToShader.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
		barrierToShader.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		barrierToShader.setImage(texRes.image);
		barrierToShader.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		barrierToShader.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
		barrierToShader.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

		cmd.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{}, nullptr, nullptr, barrierToShader
		);

		cmd.end();

		vk::SubmitInfo submit{};
		submit.setCommandBuffers(cmd);
		graphicsQueue.submit(submit, nullptr);
		graphicsQueue.waitIdle();

		device.destroyCommandPool(cmdPool);
		vmaDestroyBuffer(allocator, stagingBuffer, stagingAlloc);

		// 4. Create Image View
		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.setImage(texRes.image);
		viewInfo.setViewType(vk::ImageViewType::e2D);
		viewInfo.setFormat(format);
		viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		texRes.imageView = device.createImageView(viewInfo);

		int32_t globalIdx = static_cast<int32_t>(textures.size());
		texRes.globalIndex = globalIdx;
		textures.push_back(texRes);

		// 5. Update Descriptor Set
		vk::DescriptorImageInfo imgDesc{};
		imgDesc.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		imgDesc.setImageView(texRes.imageView);
		imgDesc.setSampler(sampler);

		vk::WriteDescriptorSet write{};
		write.setDstSet(descriptorSet);
		write.setDstBinding(0);
		write.setDstArrayElement(globalIdx);
		write.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		write.setImageInfo(imgDesc);

		device.updateDescriptorSets(write, nullptr);

		return globalIdx;
	}

	int32_t GltfTextureManager::LoadGltfImage(
		const fastgltf::Asset& asset,
		const fastgltf::Image& image,
		const std::string& baseDir
	) {
		int width = 0, height = 0, channels = 0;
		stbi_uc* pixels = nullptr;

		std::visit(
			fastgltf::visitor{
				[&](const fastgltf::sources::BufferView& bv) {
					const auto& view = asset.bufferViews[bv.bufferViewIndex];
					const auto& buffer = asset.buffers[view.bufferIndex];
					std::visit(
						fastgltf::visitor{
							[&](const fastgltf::sources::Array& array) {
								pixels = stbi_load_from_memory(
									reinterpret_cast<const stbi_uc*>(array.bytes.data() + view.byteOffset),
									static_cast<int>(view.byteLength),
									&width, &height, &channels, 4
								);
							},
							[&](const fastgltf::sources::Vector& vec) {
								pixels = stbi_load_from_memory(
									reinterpret_cast<const stbi_uc*>(vec.bytes.data() + view.byteOffset),
									static_cast<int>(view.byteLength),
									&width, &height, &channels, 4
								);
							},
							[](const auto&) {}
						},
						buffer.data
					);
				},
				[&](const fastgltf::sources::URI& uri) {
					std::string fullPath = (std::filesystem::path(baseDir) / uri.uri.path()).string();
					pixels = stbi_load(fullPath.c_str(), &width, &height, &channels, 4);
				},
				[&](const fastgltf::sources::Vector& vec) {
					pixels = stbi_load_from_memory(
						reinterpret_cast<const stbi_uc*>(vec.bytes.data()),
						static_cast<int>(vec.bytes.size()),
						&width, &height, &channels, 4
					);
				},
				[](const auto&) {}
			},
			image.data
		);

		if (!pixels) {
			return 0; // white fallback
		}

		int32_t idx = CreateTextureFromPixels(pixels, width, height, 4, vk::Format::eR8G8B8A8Unorm);
		stbi_image_free(pixels);
		return idx;
	}

} // namespace brassica
