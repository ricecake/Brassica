#include "terrain/TerrainClipmap.hpp"

#include <algorithm>
#include <cmath>

#include "spdlog/spdlog.h"
#include <FastNoise/FastNoise.h>

#include "terrain/AsyncTerrainUploader.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	struct TerrainNoiseGenerators {
		FastNoise::SmartNode<> baseScale;
		FastNoise::SmartNode<> detailScale;
		FastNoise::SmartNode<> maskScale;
		FastNoise::SmartNode<> biomeScale;
		FastNoise::SmartNode<> land;

		TerrainNoiseGenerators() {
			land = FastNoise::NewFromEncodedNodeTree(
				"FQkpCQ4JFgIXCRkJBgAAQBxGDAITCR@BD6RDAB@BCQs@AB6RBL/@AEA5qZE0IEAwrXoz4K/wEADAr/"
				"BAAEAv8EAAQD@CBMAAEjCGwAAekQEAh0JFAkQCQYMC+xRuD4EAg0JCwAAgLNCEKRwPb8YmpmZPyAC@BOAQ@BMC65HYT4M"
			);
			biomeScale = FastNoise::NewFromEncodedNodeTree("D4AF34S9AQAAAHgA");
			maskScale = FastNoise::NewFromEncodedNodeTree("D4AF34S9AQAAAHgA");
		}
	};

	static TerrainNoiseGenerators& GetGenerators() {
		static TerrainNoiseGenerators gens;
		return gens;
	}

	static float EvalHeightFromComponents(float baseVal, float detailVal, float maskVal, float biomeVal) {
		float biome = std::clamp(biomeVal, 0.0f, 1.0f);

		float mountainFactor = std::clamp((maskVal - (-0.1f)) / 0.4f, 0.0f, 1.0f);
		mountainFactor = mountainFactor * mountainFactor * (3.0f - 2.0f * mountainFactor);
		float detailFactor = (0.15f + 0.85f * mountainFactor) * std::lerp(0.3f, 1.2f, biome);

		float baseOffset = std::lerp(-35.0f, 30.0f, biome);
		float baseHeightScale = std::lerp(40.0f, 120.0f, biome);
		float detailHeightScale = std::lerp(20.0f, 80.0f, biome);

		return baseOffset + baseVal * baseHeightScale + detailVal * detailHeightScale * detailFactor;
	}

	glm::vec4 TerrainClipmap::SampleTerrain(float worldX, float worldZ, float texelSize) {
		auto&         gens = GetGenerators();
		constexpr int seed = 1337;
		float         eps = std::max(0.25f, texelSize);

		auto evalHeight = [&](float x, float z) {
			float maskVal = gens.maskScale ? gens.maskScale->GenSingle2D(x, z, seed) : 0.0f;
			float baseVal = gens.baseScale ? gens.baseScale->GenSingle2D(x, z, seed) : 0.0f;
			float detailVal = gens.detailScale ? gens.detailScale->GenSingle2D(x, z, seed) : 0.0f;
			float biomeVal = gens.biomeScale ? gens.biomeScale->GenSingle2D(x, z, seed) : 0.0f;

			return EvalHeightFromComponents(baseVal, detailVal, maskVal, biomeVal);
		};

		float h = evalHeight(worldX, worldZ);
		float hL = evalHeight(worldX - eps, worldZ);
		float hR = evalHeight(worldX + eps, worldZ);
		float hD = evalHeight(worldX, worldZ - eps);
		float hU = evalHeight(worldX, worldZ + eps);

		glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));
		return glm::vec4(h, normal.x, normal.y, normal.z);
	}

	struct TerrainRegionData {
		std::vector<glm::vec4> heightMap;
		std::vector<glm::vec4> minMaxMap;
		std::vector<glm::vec4> biomeMap;
		std::vector<glm::vec4> visibilityMap;
	};

	static TerrainRegionData GenerateTerrainRegionData(
		const ClipmapLevelInfo& info,
		uint32_t                startX,
		uint32_t                startZ,
		uint32_t                width,
		uint32_t                height,
		int                     deltaX = 0,
		int                     deltaZ = 0
	) {
		TerrainRegionData data;
		size_t            size = static_cast<size_t>(width) * height;
		data.heightMap.resize(size);
		data.minMaxMap.resize(size);
		data.biomeMap.resize(size);
		data.visibilityMap.resize(size);

		float texelSize = info.texelSize > 0.0f ? info.texelSize : 0.5f;
		float halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

		float minWorldX = info.centerWorldPos.x - halfExtent;
		float minWorldZ = info.centerWorldPos.y - halfExtent;

		if (deltaX != 0) {
			if (deltaX > 0) {
				minWorldX = info.centerWorldPos.x + halfExtent - static_cast<float>(width) * texelSize;
			} else {
				minWorldX = info.centerWorldPos.x - halfExtent;
			}
		} else if (width < TERRAIN_MAP_DIM) {
			int colIdx = (static_cast<int>(startX) - info.gridOffset.x + static_cast<int>(TERRAIN_MAP_DIM)) %
				static_cast<int>(TERRAIN_MAP_DIM);
			minWorldX = info.centerWorldPos.x - halfExtent + static_cast<float>(colIdx) * texelSize;
		}

		if (deltaZ != 0) {
			if (deltaZ > 0) {
				minWorldZ = info.centerWorldPos.y + halfExtent - static_cast<float>(height) * texelSize;
			} else {
				minWorldZ = info.centerWorldPos.y - halfExtent;
			}
		} else if (height < TERRAIN_MAP_DIM) {
			int rowIdx = (static_cast<int>(startZ) - info.gridOffset.y + static_cast<int>(TERRAIN_MAP_DIM)) %
				static_cast<int>(TERRAIN_MAP_DIM);
			minWorldZ = info.centerWorldPos.y - halfExtent + static_cast<float>(rowIdx) * texelSize;
		}

		uint32_t paddedW = width + 2;
		uint32_t paddedH = height + 2;
		size_t   totalPadded = static_cast<size_t>(paddedW) * paddedH;

		std::vector<float> paddedHeights(totalPadded, 0.0f);
		std::vector<float> biomePatch(totalPadded, 0.5f);
		std::vector<float> maskPatch(totalPadded, 0.0f);

		auto&         gens = GetGenerators();
		constexpr int seed = 1337;

		float gridStartX = minWorldX - texelSize;
		float gridStartZ = minWorldZ - texelSize;

		gens.land->GenUniformGrid2D(
			paddedHeights.data(),
			gridStartX,
			gridStartZ,
			paddedW,
			paddedH,
			texelSize,
			texelSize,
			seed
		);

		if (gens.biomeScale) {
			gens.biomeScale->GenUniformGrid2D(
				biomePatch.data(),
				gridStartX,
				gridStartZ,
				paddedW,
				paddedH,
				texelSize,
				texelSize,
				seed
			);
		}

		if (gens.maskScale) {
			gens.maskScale->GenUniformGrid2D(
				maskPatch.data(),
				gridStartX,
				gridStartZ,
				paddedW,
				paddedH,
				texelSize,
				texelSize,
				seed
			);
		}

		float eps = std::max(0.25f, texelSize);

		for (uint32_t z = 0; z < height; ++z) {
			size_t rowIdx = static_cast<size_t>(z + 1) * paddedW;
			size_t prevRow = static_cast<size_t>(z) * paddedW;
			size_t nextRow = static_cast<size_t>(z + 2) * paddedW;

			uint32_t destZ = (height == TERRAIN_MAP_DIM) ? ((z + info.gridOffset.y) % TERRAIN_MAP_DIM) : z;

			for (uint32_t x = 0; x < width; ++x) {
				size_t colIdx = x + 1;
				float  h = paddedHeights[rowIdx + colIdx];
				float  hL = paddedHeights[rowIdx + x];
				float  hR = paddedHeights[rowIdx + x + 2];
				float  hD = paddedHeights[prevRow + colIdx];
				float  hU = paddedHeights[nextRow + colIdx];

				float hLD = paddedHeights[prevRow + x];
				float hRD = paddedHeights[prevRow + x + 2];
				float hLU = paddedHeights[nextRow + x];
				float hRU = paddedHeights[nextRow + x + 2];

				float minH = std::min({h, hL, hR, hD, hU, hLD, hRD, hLU, hRU});
				float maxH = std::max({h, hL, hR, hD, hU, hLD, hRD, hLU, hRU});
				float variance = maxH - minH;

				float biomeVal = std::clamp(biomePatch[rowIdx + colIdx], 0.0f, 1.0f);
				float maskVal = maskPatch[rowIdx + colIdx];

				glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));

				uint32_t destX = (width == TERRAIN_MAP_DIM) ? ((x + info.gridOffset.x) % TERRAIN_MAP_DIM) : x;
				size_t   idx = destZ * width + destX;

				data.heightMap[idx] = glm::vec4(h, normal.x, normal.y, normal.z);
				data.minMaxMap[idx] = glm::vec4(minH, maxH, minH, maxH);
				data.biomeMap[idx] = glm::vec4(biomeVal, variance, 1.0f, maskVal);
				data.visibilityMap[idx] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}

		return data;
	}

	TerrainClipmap::~TerrainClipmap() {
		Cleanup();
	}

	void TerrainClipmap::Init(
		vk::Device       dev,
		VmaAllocator     alloc,
		uint32_t         lods,
		float            baseTexel,
		float            maxDist,
		const glm::vec3& initialCameraPos
	) {
		device = dev;
		allocator = alloc;
		baseTexelSize = baseTexel;

		if (maxDist > 0.0f) {
			float    baseLevelExtent = static_cast<float>(TERRAIN_MAP_DIM) * baseTexelSize;
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
			levelInfos[i].centerWorldPos = glm::floor(
											   glm::vec2(initialCameraPos.x, initialCameraPos.z) /
											   levelInfos[i].texelSize
										   ) *
				levelInfos[i].texelSize;
			levelInfos[i].gridOffset = glm::ivec2(0);
		}

		CreateTextureArrays();
		CreateSampler();
	}

	void
	TerrainClipmap::UpdateCameraPosition(const glm::vec3& cameraPos, AsyncTerrainUploader& uploader, vk::Queue queue) {
		for (uint32_t l = 0; l < numLODs; ++l) {
			auto& info = levelInfos[l];
			float texelSize = info.texelSize;

			glm::vec2 newCenter = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / texelSize) * texelSize;
			glm::vec2 diff = newCenter - info.centerWorldPos;

			int deltaX = static_cast<int>(std::round(diff.x / texelSize));
			int deltaZ = static_cast<int>(std::round(diff.y / texelSize));

			if (deltaX == 0 && deltaZ == 0)
				continue;

			if (std::abs(deltaX) >= static_cast<int>(TERRAIN_MAP_DIM) ||
			    std::abs(deltaZ) >= static_cast<int>(TERRAIN_MAP_DIM)) {
				info.centerWorldPos = newCenter;
				info.gridOffset = glm::ivec2(0);
				auto levelData = GenerateLevelData(l);
				uploader.UploadLevelAsync(l, levelData.heightMap, image, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
				uploader.UploadLevelAsync(l, levelData.minMaxMap, minmaxImage, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
				uploader.UploadLevelAsync(l, levelData.biomeMap, biomeImage, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
				uploader.UploadLevelAsync(
					l,
					levelData.visibilityMap,
					visibilityImage,
					TERRAIN_MAP_DIM,
					TERRAIN_MAP_DIM,
					queue
				);
				continue;
			}

			info.centerWorldPos = newCenter;

			std::vector<glm::vec4>           updateHeightBuffer;
			std::vector<glm::vec4>           updateMinMaxBuffer;
			std::vector<glm::vec4>           updateBiomeBuffer;
			std::vector<glm::vec4>           updateVisibilityBuffer;
			std::vector<vk::BufferImageCopy> copyRegions;

			if (deltaX != 0) {
				uint32_t stripWidth = std::abs(deltaX);
				int      startDstX = (deltaX > 0) ? info.gridOffset.x
												  : ((info.gridOffset.x + deltaX + static_cast<int>(TERRAIN_MAP_DIM)) %
												     static_cast<int>(TERRAIN_MAP_DIM));

				TerrainRegionData stripData =
					GenerateTerrainRegionData(info, 0, 0, stripWidth, TERRAIN_MAP_DIM, deltaX, 0);

				size_t baseOffset = updateHeightBuffer.size() * sizeof(glm::vec4);
				updateHeightBuffer.insert(updateHeightBuffer.end(), stripData.heightMap.begin(), stripData.heightMap.end());
				updateMinMaxBuffer.insert(updateMinMaxBuffer.end(), stripData.minMaxMap.begin(), stripData.minMaxMap.end());
				updateBiomeBuffer.insert(updateBiomeBuffer.end(), stripData.biomeMap.begin(), stripData.biomeMap.end());
				updateVisibilityBuffer.insert(
					updateVisibilityBuffer.end(),
					stripData.visibilityMap.begin(),
					stripData.visibilityMap.end()
				);

				if (startDstX + stripWidth <= TERRAIN_MAP_DIM) {
					vk::BufferImageCopy copyRegion{};
					copyRegion.setBufferOffset(baseOffset);
					copyRegion.setBufferRowLength(stripWidth);
					copyRegion.setBufferImageHeight(TERRAIN_MAP_DIM);
					copyRegion.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
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
					copyRegion1.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
					copyRegion1.setImageOffset(vk::Offset3D{startDstX, 0, 0});
					copyRegion1.setImageExtent(vk::Extent3D{w1, TERRAIN_MAP_DIM, 1});

					vk::BufferImageCopy copyRegion2{};
					copyRegion2.setBufferOffset(baseOffset + static_cast<size_t>(w1) * sizeof(glm::vec4));
					copyRegion2.setBufferRowLength(stripWidth);
					copyRegion2.setBufferImageHeight(TERRAIN_MAP_DIM);
					copyRegion2.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
					copyRegion2.setImageOffset(vk::Offset3D{0, 0, 0});
					copyRegion2.setImageExtent(vk::Extent3D{w2, TERRAIN_MAP_DIM, 1});

					copyRegions.push_back(copyRegion1);
					copyRegions.push_back(copyRegion2);
				}

				info.gridOffset.x = (info.gridOffset.x + deltaX) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.x < 0)
					info.gridOffset.x += static_cast<int>(TERRAIN_MAP_DIM);
			}

			if (deltaZ != 0) {
				uint32_t stripHeight = std::abs(deltaZ);
				int      startDstZ = (deltaZ > 0) ? info.gridOffset.y
												  : ((info.gridOffset.y + deltaZ + static_cast<int>(TERRAIN_MAP_DIM)) %
												     static_cast<int>(TERRAIN_MAP_DIM));

				TerrainRegionData stripData =
					GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, stripHeight, 0, deltaZ);

				size_t baseOffset = updateHeightBuffer.size() * sizeof(glm::vec4);
				updateHeightBuffer.insert(updateHeightBuffer.end(), stripData.heightMap.begin(), stripData.heightMap.end());
				updateMinMaxBuffer.insert(updateMinMaxBuffer.end(), stripData.minMaxMap.begin(), stripData.minMaxMap.end());
				updateBiomeBuffer.insert(updateBiomeBuffer.end(), stripData.biomeMap.begin(), stripData.biomeMap.end());
				updateVisibilityBuffer.insert(
					updateVisibilityBuffer.end(),
					stripData.visibilityMap.begin(),
					stripData.visibilityMap.end()
				);

				if (startDstZ + stripHeight <= TERRAIN_MAP_DIM) {
					vk::BufferImageCopy copyRegion{};
					copyRegion.setBufferOffset(baseOffset);
					copyRegion.setBufferRowLength(TERRAIN_MAP_DIM);
					copyRegion.setBufferImageHeight(stripHeight);
					copyRegion.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
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
					copyRegion1.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
					copyRegion1.setImageOffset(vk::Offset3D{0, startDstZ, 0});
					copyRegion1.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, h1, 1});

					vk::BufferImageCopy copyRegion2{};
					copyRegion2.setBufferOffset(
						baseOffset + static_cast<size_t>(h1) * TERRAIN_MAP_DIM * sizeof(glm::vec4)
					);
					copyRegion2.setBufferRowLength(TERRAIN_MAP_DIM);
					copyRegion2.setBufferImageHeight(stripHeight);
					copyRegion2.setImageSubresource(
						vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, l, 1)
					);
					copyRegion2.setImageOffset(vk::Offset3D{0, 0, 0});
					copyRegion2.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, h2, 1});

					copyRegions.push_back(copyRegion1);
					copyRegions.push_back(copyRegion2);
				}

				info.gridOffset.y = (info.gridOffset.y + deltaZ) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.y < 0)
					info.gridOffset.y += static_cast<int>(TERRAIN_MAP_DIM);
			}

			if (!updateHeightBuffer.empty() && !copyRegions.empty()) {
				uploader.UploadRegionAsync(l, updateHeightBuffer, copyRegions, image, queue);
				uploader.UploadRegionAsync(l, updateMinMaxBuffer, copyRegions, minmaxImage, queue);
				uploader.UploadRegionAsync(l, updateBiomeBuffer, copyRegions, biomeImage, queue);
				uploader.UploadRegionAsync(l, updateVisibilityBuffer, copyRegions, visibilityImage, queue);
			}
		}
	}

	void TerrainClipmap::Cleanup() {
		if (device) {
			if (sampler) {
				device.destroySampler(sampler);
				sampler = nullptr;
			}
			auto destroyArrayImage = [&](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
				if (view) {
					device.destroyImageView(view);
					view = nullptr;
				}
				if (img && alloc && allocator != VK_NULL_HANDLE) {
					vmaDestroyImage(allocator, img, alloc);
					img = nullptr;
					alloc = VK_NULL_HANDLE;
				}
			};
			destroyArrayImage(image, imageView, allocation);
			destroyArrayImage(minmaxImage, minmaxImageView, minmaxAllocation);
			destroyArrayImage(biomeImage, biomeImageView, biomeAllocation);
			destroyArrayImage(visibilityImage, visibilityImageView, visibilityAllocation);
		}
	}

	void TerrainClipmap::CreateTextureArrays() {
		auto createArrayImage = [&](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
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
			if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &alloc, nullptr) != VK_SUCCESS) {
				spdlog::error("Failed to create TerrainClipmap texture array image!");
				return;
			}
			img = vkImg;

			vk::ImageViewCreateInfo viewInfo{};
			viewInfo.setImage(img);
			viewInfo.setViewType(vk::ImageViewType::e2DArray);
			viewInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
			viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, numLODs));
			view = device.createImageView(viewInfo);
		};

		createArrayImage(image, imageView, allocation);
		createArrayImage(minmaxImage, minmaxImageView, minmaxAllocation);
		createArrayImage(biomeImage, biomeImageView, biomeAllocation);
		createArrayImage(visibilityImage, visibilityImageView, visibilityAllocation);
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

	std::vector<glm::vec4> TerrainClipmap::GenerateLevelMap(uint32_t levelIndex) const {
		ClipmapLevelInfo info{};
		if (levelIndex < levelInfos.size()) {
			info = levelInfos[levelIndex];
		} else {
			info.level = levelIndex;
			info.baseTexelSize = baseTexelSize > 0.0f ? baseTexelSize : 0.5f;
			info.texelSize = info.baseTexelSize * static_cast<float>(1 << levelIndex);
			info.worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * info.texelSize;
			info.centerWorldPos = glm::vec2(0.0f);
			info.gridOffset = glm::ivec2(0);
		}
		return GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0).heightMap;
	}

	TerrainClipmap::TerrainLevelData TerrainClipmap::GenerateLevelData(uint32_t levelIndex) const {
		ClipmapLevelInfo info{};
		if (levelIndex < levelInfos.size()) {
			info = levelInfos[levelIndex];
		} else {
			info.level = levelIndex;
			info.baseTexelSize = baseTexelSize > 0.0f ? baseTexelSize : 0.5f;
			info.texelSize = info.baseTexelSize * static_cast<float>(1 << levelIndex);
			info.worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * info.texelSize;
			info.centerWorldPos = glm::vec2(0.0f);
			info.gridOffset = glm::ivec2(0);
		}
		TerrainRegionData region =
			GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0);
		return TerrainLevelData{
			.heightMap = std::move(region.heightMap),
			.minMaxMap = std::move(region.minMaxMap),
			.biomeMap = std::move(region.biomeMap),
			.visibilityMap = std::move(region.visibilityMap)
		};
	}

	std::vector<glm::vec4> TerrainClipmap::GenerateSineWaveMap(
		uint32_t         levelIndex,
		float            baseTexelSize,
		const glm::vec2& centerWorldPos,
		float            time
	) {
		ClipmapLevelInfo info{};
		info.level = levelIndex;
		info.baseTexelSize = baseTexelSize;
		info.texelSize = baseTexelSize * static_cast<float>(1 << levelIndex);
		info.worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * info.texelSize;
		info.centerWorldPos = centerWorldPos;
		info.gridOffset = glm::ivec2(0);

		return GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0).heightMap;
	}

} // namespace brassica
