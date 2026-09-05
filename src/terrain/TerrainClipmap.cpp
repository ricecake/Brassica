#include "terrain/TerrainClipmap.hpp"
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "spdlog/spdlog.h"

namespace brassica {

	TerrainClipmap::~TerrainClipmap() {
		Cleanup();
	}

	void TerrainClipmap::Init(vk::Device dev, VmaAllocator alloc, uint32_t lods, float baseTexel) {
		device = dev;
		allocator = alloc;
		numLODs = lods;
		baseTexelSize = baseTexel;

		levelInfos.resize(numLODs);
		for (uint32_t i = 0; i < numLODs; ++i) {
			levelInfos[i].level = i;
			levelInfos[i].baseTexelSize = baseTexelSize;
			levelInfos[i].texelSize = baseTexelSize * static_cast<float>(1 << i);
			levelInfos[i].worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * levelInfos[i].texelSize;
			levelInfos[i].centerWorldPos = glm::vec2(0.0f);
		}

		CreateTextureArray();
		CreateSampler();
	}

	void TerrainClipmap::Cleanup() {
		if (device) {
			if (sampler) {
				device.destroySampler(sampler);
				sampler = nullptr;
			}
			if (imageView) {
				device.destroyImageView(imageView);
				imageView = nullptr;
			}
			if (image && allocation && allocator != VK_NULL_HANDLE) {
				vmaDestroyImage(allocator, image, allocation);
				image = nullptr;
				allocation = VK_NULL_HANDLE;
			}
		}
	}

	void TerrainClipmap::CreateTextureArray() {
		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = VkExtent3D{TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 1};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = numLODs;
		imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		VkImage vkImg = VK_NULL_HANDLE;
		if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &allocation, nullptr) != VK_SUCCESS) {
			spdlog::error("Failed to create TerrainClipmap texture array image!");
			return;
		}
		image = vkImg;

		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.setImage(image);
		viewInfo.setViewType(vk::ImageViewType::e2DArray);
		viewInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
		viewInfo.setSubresourceRange(
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, numLODs)
		);
		imageView = device.createImageView(viewInfo);
	}

	void TerrainClipmap::CreateSampler() {
		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.setMagFilter(vk::Filter::eLinear);
		samplerInfo.setMinFilter(vk::Filter::eLinear);
		samplerInfo.setMipmapMode(vk::SamplerMipmapMode::eLinear);
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eClampToEdge);
		samplerInfo.setAnisotropyEnable(VK_FALSE);
		samplerInfo.setMaxAnisotropy(1.0f);
		samplerInfo.setBorderColor(vk::BorderColor::eFloatOpaqueBlack);
		samplerInfo.setUnnormalizedCoordinates(VK_FALSE);

		sampler = device.createSampler(samplerInfo);
	}

	std::vector<glm::vec4> TerrainClipmap::GenerateSineWaveMap(
		uint32_t levelIndex,
		float baseTexelSize,
		const glm::vec2& centerWorldPos,
		float time
	) {
		std::vector<glm::vec4> data(TERRAIN_MAP_DIM * TERRAIN_MAP_DIM);
		float texelSize = baseTexelSize * static_cast<float>(1 << levelIndex);
		float halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

		auto heightFunc = [time](float x, float z) -> float {
			float wave1 = std::sin(0.05f * x + time) * 2.5f;
			float wave2 = std::cos(0.05f * z + time * 0.8f) * 2.5f;
			float wave3 = std::sin(0.02f * (x + z)) * 1.5f;
			return wave1 + wave2 + wave3;
		};

		for (uint32_t z = 0; z < TERRAIN_MAP_DIM; ++z) {
			for (uint32_t x = 0; x < TERRAIN_MAP_DIM; ++x) {
				float worldX = centerWorldPos.x - halfExtent + static_cast<float>(x) * texelSize;
				float worldZ = centerWorldPos.y - halfExtent + static_cast<float>(z) * texelSize;

				float h = heightFunc(worldX, worldZ);

				// Compute analytical / central difference normals
				float eps = texelSize;
				float hL = heightFunc(worldX - eps, worldZ);
				float hR = heightFunc(worldX + eps, worldZ);
				float hD = heightFunc(worldX, worldZ - eps);
				float hU = heightFunc(worldX, worldZ + eps);

				glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));

				data[z * TERRAIN_MAP_DIM + x] = glm::vec4(h, normal.x, normal.y, normal.z);
			}
		}

		return data;
	}

} // namespace brassica
