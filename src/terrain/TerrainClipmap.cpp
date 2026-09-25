#include "terrain/TerrainClipmap.hpp"

#include <algorithm>
#include <cmath>

#include "spdlog/spdlog.h"
#include <FastNoise/FastNoise.h>

#include "EngineConstants.hpp"
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
				// "KQkdCRMABg@BkSCS4AAQ@BkHAABAnEUUCw@AD8EAgsAAECcRS@DOAE@BEC83MTD4EAw@CL@BPxMAAEjCGwAAekQE"
				// "KQkdCRMABg@BkSCS4AAQ@BkHAABAnEUUCw@AD8EAgsAAECcRS@DOAE@BEC83MTD4EAw@CT@AgwRsAAHpEBA=="
				//  "KQkdCRMABg@BkSCS4AAQ@BkHAABAnEUUCw@AD8EAgsAAECcRS@DOAE@BEC83MTD4EEwAAIMEbAAD6QwQ="
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
		glm::vec2 wrapped = WrapShortestDistance(glm::vec2(worldX, worldZ));
		worldX = wrapped.x;
		worldZ = wrapped.y;
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
			uint32_t derivedLODs = 1;
			while (derivedLODs < constants::Class::Terrain::DefaultMaxLODs &&
			       (static_cast<float>(TERRAIN_MAP_DIM) * GetLODScale(static_cast<float>(derivedLODs - 1)) *
			        baseTexelSize * 0.5f) < maxDist) {
				derivedLODs++;
			}
			numLODs = std::clamp(derivedLODs, 1u, static_cast<uint32_t>(constants::Class::Terrain::DefaultMaxLODs));
		} else {
			numLODs = lods;
		}

		levelInfos.resize(numLODs);
		for (uint32_t i = 0; i < numLODs; ++i) {
			levelInfos[i].level = i;
			levelInfos[i].baseTexelSize = baseTexelSize;
			levelInfos[i].texelSize = baseTexelSize * GetLODScale(static_cast<float>(i));
			levelInfos[i].worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * levelInfos[i].texelSize;
			levelInfos[i].centerWorldPos = glm::floor(
											   glm::vec2(initialCameraPos.x, initialCameraPos.z) /
											   levelInfos[i].texelSize
										   ) *
				levelInfos[i].texelSize;
			levelInfos[i].gridOffset = glm::ivec2(0);
			levelInfos[i].delta = glm::ivec2(TERRAIN_MAP_DIM, TERRAIN_MAP_DIM);
		}

		CreateTextureArrays();
		CreateSampler();
	}

	void TerrainClipmap::UpdateCameraPosition(const glm::vec3& cameraPos, const glm::vec3& wrapOffset) {
		for (uint32_t l = 0; l < numLODs; ++l) {
			auto& info = levelInfos[l];
			float texelSize = info.texelSize;

			info.centerWorldPos += glm::vec2(wrapOffset.x, wrapOffset.z);

			glm::vec2 newCenter = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / texelSize) * texelSize;
			glm::vec2 diff = newCenter - info.centerWorldPos;

			int deltaX = static_cast<int>(std::round(diff.x / texelSize));
			int deltaZ = static_cast<int>(std::round(diff.y / texelSize));

			if (m_forceRegenerate) {
				info.delta = glm::ivec2(TERRAIN_MAP_DIM, TERRAIN_MAP_DIM);
				info.centerWorldPos = newCenter;
				info.gridOffset.x = (info.gridOffset.x + deltaX) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.x < 0)
					info.gridOffset.x += static_cast<int>(TERRAIN_MAP_DIM);

				info.gridOffset.y = (info.gridOffset.y + deltaZ) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.y < 0)
					info.gridOffset.y += static_cast<int>(TERRAIN_MAP_DIM);
			} else if (deltaX == 0 && deltaZ == 0) {
				info.delta = glm::ivec2(0, 0);
				continue;
			} else {
				info.delta = glm::ivec2(deltaX, deltaZ);
				info.centerWorldPos = newCenter;
				info.gridOffset.x = (info.gridOffset.x + deltaX) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.x < 0)
					info.gridOffset.x += static_cast<int>(TERRAIN_MAP_DIM);

				info.gridOffset.y = (info.gridOffset.y + deltaZ) % static_cast<int>(TERRAIN_MAP_DIM);
				if (info.gridOffset.y < 0)
					info.gridOffset.y += static_cast<int>(TERRAIN_MAP_DIM);
			}
		}
		m_forceRegenerate = false;
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
			imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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
		return GenerateTerrainRegionData(levelInfos[levelIndex], 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0)
			.heightMap;
	}

	TerrainClipmap::TerrainLevelData TerrainClipmap::GenerateLevelData(uint32_t levelIndex) const {
		ClipmapLevelInfo info{};
		if (levelIndex < levelInfos.size()) {
			info = levelInfos[levelIndex];
		} else {
			info.level = levelIndex;
			info.baseTexelSize = baseTexelSize > 0.0f ? baseTexelSize : 0.5f;
			info.texelSize = info.baseTexelSize * GetLODScale(static_cast<float>(levelIndex));
			info.worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * info.texelSize;
			info.centerWorldPos = glm::vec2(0.0f);
			info.gridOffset = glm::ivec2(0);
		}
		TerrainRegionData region = GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0);
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
		info.texelSize = baseTexelSize * GetLODScale(static_cast<float>(levelIndex));
		info.worldExtent = static_cast<float>(TERRAIN_MAP_DIM) * info.texelSize;
		info.centerWorldPos = centerWorldPos;
		info.gridOffset = glm::ivec2(0);

		return GenerateTerrainRegionData(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0).heightMap;
	}

} // namespace brassica
