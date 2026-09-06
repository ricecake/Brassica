#include "terrain/TerrainClipmap.hpp"
#include "terrain/AsyncTerrainUploader.hpp"
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include "spdlog/spdlog.h"

namespace brassica {

	TerrainClipmap::~TerrainClipmap() {
		Cleanup();
	}

	void TerrainClipmap::Init(vk::Device dev, VmaAllocator alloc, uint32_t lods, float baseTexel, float maxDist) {
		device = dev;
		allocator = alloc;
		baseTexelSize = baseTexel;

		if (maxDist > 0.0f) {
			float baseLevelExtent = static_cast<float>(TERRAIN_MAP_DIM) * baseTexelSize;
			uint32_t derivedLODs = static_cast<uint32_t>(std::ceil(std::log2(maxDist / baseLevelExtent))) + 1;
			numLODs = std::clamp(derivedLODs, 1u, 8u);
		} else {
			numLODs = lods;
		}

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

	void TerrainClipmap::UpdateCameraPosition(const glm::vec3& cameraPos, AsyncTerrainUploader& uploader, vk::Queue queue) {
		for (uint32_t l = 0; l < numLODs; ++l) {
			auto& info = levelInfos[l];
			float texelSize = info.texelSize;

			glm::vec2 newCenter = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / texelSize) * texelSize;
			glm::vec2 diff = newCenter - info.centerWorldPos;

			int deltaX = static_cast<int>(std::round(diff.x / texelSize));
			int deltaZ = static_cast<int>(std::round(diff.y / texelSize));

			if (deltaX == 0 && deltaZ == 0) continue;

			if (std::abs(deltaX) >= static_cast<int>(TERRAIN_MAP_DIM) || std::abs(deltaZ) >= static_cast<int>(TERRAIN_MAP_DIM)) {
				info.centerWorldPos = newCenter;
				info.gridOffset = glm::ivec2(0);
				auto mapData = GenerateSineWaveMap(l, baseTexelSize, info.centerWorldPos);
				uploader.UploadLevelAsync(l, mapData, image, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
				continue;
			}

			info.centerWorldPos = newCenter;

			std::vector<glm::vec4> updateBuffer;
			std::vector<vk::BufferImageCopy> copyRegions;

			if (deltaX != 0) {
				uint32_t stripWidth = std::abs(deltaX);
				int startDstX = (deltaX > 0)
					? info.gridOffset.x
					: ((info.gridOffset.x + deltaX + static_cast<int>(TERRAIN_MAP_DIM)) % static_cast<int>(TERRAIN_MAP_DIM));

				std::vector<glm::vec4> stripData(stripWidth * TERRAIN_MAP_DIM);
				float halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

				auto heightFunc = [](float x, float z) -> float {
					float wave1 = std::sin(0.05f * x) * 2.5f;
					float wave2 = std::cos(0.05f * z) * 2.5f;
					float wave3 = std::sin(0.02f * (x + z)) * 1.5f;
					return wave1 + wave2 + wave3;
				};

				for (uint32_t z = 0; z < TERRAIN_MAP_DIM; ++z) {
					int localGridZ = (static_cast<int>(z) - info.gridOffset.y + static_cast<int>(TERRAIN_MAP_DIM)) % static_cast<int>(TERRAIN_MAP_DIM);
					float worldZ = info.centerWorldPos.y - halfExtent + static_cast<float>(localGridZ) * texelSize;

					for (uint32_t x = 0; x < stripWidth; ++x) {
						int colIdx = (deltaX > 0) ? (TERRAIN_MAP_DIM - stripWidth + x) : x;
						float worldX = info.centerWorldPos.x - halfExtent + static_cast<float>(colIdx) * texelSize;

						float h = heightFunc(worldX, worldZ);
						float eps = texelSize;
						float hL = heightFunc(worldX - eps, worldZ);
						float hR = heightFunc(worldX + eps, worldZ);
						float hD = heightFunc(worldX, worldZ - eps);
						float hU = heightFunc(worldX, worldZ + eps);

						glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));
						stripData[z * stripWidth + x] = glm::vec4(h, normal.x, normal.y, normal.z);
					}
				}

				size_t baseOffset = updateBuffer.size() * sizeof(glm::vec4);
				updateBuffer.insert(updateBuffer.end(), stripData.begin(), stripData.end());

				if (startDstX + stripWidth <= TERRAIN_MAP_DIM) {
					vk::BufferImageCopy copyRegion{};
					copyRegion.setBufferOffset(baseOffset);
					copyRegion.setBufferRowLength(stripWidth);
					copyRegion.setBufferImageHeight(TERRAIN_MAP_DIM);
					copyRegion.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion.setImageOffset(vk::Offset3D{startDstX, 0, 0});
					copyRegion.setImageExtent(vk::Extent3D{stripWidth, TERRAIN_MAP_DIM, 1});

					copyRegions.push_back(copyRegion);
				} else {
					uint32_t w1 = TERRAIN_MAP_DIM - startDstX;
					uint32_t w2 = stripWidth - w1;

					vk::BufferImageCopy copyRegion1{};
					copyRegion1.setBufferOffset(baseOffset);
					copyRegion1.setBufferRowLength(stripWidth);
					copyRegion1.setBufferImageHeight(TERRAIN_MAP_DIM);
					copyRegion1.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion1.setImageOffset(vk::Offset3D{startDstX, 0, 0});
					copyRegion1.setImageExtent(vk::Extent3D{w1, TERRAIN_MAP_DIM, 1});

					vk::BufferImageCopy copyRegion2{};
					copyRegion2.setBufferOffset(baseOffset + static_cast<size_t>(w1) * sizeof(glm::vec4));
					copyRegion2.setBufferRowLength(stripWidth);
					copyRegion2.setBufferImageHeight(TERRAIN_MAP_DIM);
					copyRegion2.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion2.setImageOffset(vk::Offset3D{0, 0, 0});
					copyRegion2.setImageExtent(vk::Extent3D{w2, TERRAIN_MAP_DIM, 1});

					copyRegions.push_back(copyRegion1);
					copyRegions.push_back(copyRegion2);
				}

				info.gridOffset.x = (info.gridOffset.x + deltaX) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.x < 0) info.gridOffset.x += static_cast<int>(TERRAIN_MAP_DIM);
			}

			if (deltaZ != 0) {
				uint32_t stripHeight = std::abs(deltaZ);
				int startDstZ = (deltaZ > 0)
					? info.gridOffset.y
					: ((info.gridOffset.y + deltaZ + static_cast<int>(TERRAIN_MAP_DIM)) % static_cast<int>(TERRAIN_MAP_DIM));

				std::vector<glm::vec4> stripData(TERRAIN_MAP_DIM * stripHeight);
				float halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

				auto heightFunc = [](float x, float z) -> float {
					float wave1 = std::sin(0.05f * x) * 2.5f;
					float wave2 = std::cos(0.05f * z) * 2.5f;
					float wave3 = std::sin(0.02f * (x + z)) * 1.5f;
					return wave1 + wave2 + wave3;
				};

				for (uint32_t z = 0; z < stripHeight; ++z) {
					int rowIdx = (deltaZ > 0) ? (TERRAIN_MAP_DIM - stripHeight + z) : z;
					float worldZ = info.centerWorldPos.y - halfExtent + static_cast<float>(rowIdx) * texelSize;

					for (uint32_t x = 0; x < TERRAIN_MAP_DIM; ++x) {
						int localGridX = (static_cast<int>(x) - info.gridOffset.x + static_cast<int>(TERRAIN_MAP_DIM)) % static_cast<int>(TERRAIN_MAP_DIM);
						float worldX = info.centerWorldPos.x - halfExtent + static_cast<float>(localGridX) * texelSize;

						float h = heightFunc(worldX, worldZ);
						float eps = texelSize;
						float hL = heightFunc(worldX - eps, worldZ);
						float hR = heightFunc(worldX + eps, worldZ);
						float hD = heightFunc(worldX, worldZ - eps);
						float hU = heightFunc(worldX, worldZ + eps);

						glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));
						stripData[z * TERRAIN_MAP_DIM + x] = glm::vec4(h, normal.x, normal.y, normal.z);
					}
				}

				size_t baseOffset = updateBuffer.size() * sizeof(glm::vec4);
				updateBuffer.insert(updateBuffer.end(), stripData.begin(), stripData.end());

				if (startDstZ + stripHeight <= TERRAIN_MAP_DIM) {
					vk::BufferImageCopy copyRegion{};
					copyRegion.setBufferOffset(baseOffset);
					copyRegion.setBufferRowLength(TERRAIN_MAP_DIM);
					copyRegion.setBufferImageHeight(stripHeight);
					copyRegion.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion.setImageOffset(vk::Offset3D{0, startDstZ, 0});
					copyRegion.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, stripHeight, 1});

					copyRegions.push_back(copyRegion);
				} else {
					uint32_t h1 = TERRAIN_MAP_DIM - startDstZ;
					uint32_t h2 = stripHeight - h1;

					vk::BufferImageCopy copyRegion1{};
					copyRegion1.setBufferOffset(baseOffset);
					copyRegion1.setBufferRowLength(TERRAIN_MAP_DIM);
					copyRegion1.setBufferImageHeight(stripHeight);
					copyRegion1.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion1.setImageOffset(vk::Offset3D{0, startDstZ, 0});
					copyRegion1.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, h1, 1});

					vk::BufferImageCopy copyRegion2{};
					copyRegion2.setBufferOffset(baseOffset + static_cast<size_t>(h1) * TERRAIN_MAP_DIM * sizeof(glm::vec4));
					copyRegion2.setBufferRowLength(TERRAIN_MAP_DIM);
					copyRegion2.setBufferImageHeight(stripHeight);
					copyRegion2.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1));
					copyRegion2.setImageOffset(vk::Offset3D{0, 0, 0});
					copyRegion2.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, h2, 1});

					copyRegions.push_back(copyRegion1);
					copyRegions.push_back(copyRegion2);
				}

				info.gridOffset.y = (info.gridOffset.y + deltaZ) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.y < 0) info.gridOffset.y += static_cast<int>(TERRAIN_MAP_DIM);
			}

			if (!updateBuffer.empty() && !copyRegions.empty()) {
				uploader.UploadRegionAsync(l, updateBuffer, copyRegions, image, queue);
			}
		}
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
		samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
		samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
		samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
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
