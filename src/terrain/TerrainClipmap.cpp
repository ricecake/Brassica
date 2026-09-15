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

	static const uint32_t kLodOffsets[8] = {5461, 1365, 341, 85, 21, 5, 1, 0};

	size_t TerrainClipmap::GetNodeIndex(uint32_t lod, uint32_t gridX, uint32_t gridZ) const {
		uint32_t dim = 128 >> lod;
		return kLodOffsets[lod] + gridZ * dim + gridX;
	}

	TerrainClipmap::TerrainLevelData TerrainClipmap::GenerateTileData(uint32_t lod, uint32_t gridX, uint32_t gridZ) const {
		TerrainLevelData data;
		size_t           size = static_cast<size_t>(TERRAIN_MAP_DIM) * TERRAIN_MAP_DIM;
		data.heightMap.resize(size);
		data.minMaxMap.resize(size);
		data.biomeMap.resize(size);
		data.visibilityMap.resize(size);

		float texelSize = baseTexelSize * static_cast<float>(1 << lod);
		float tileExtent = static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

		float minWorldX = ROOT_WORLD_MIN_X + static_cast<float>(gridX) * tileExtent;
		float minWorldZ = ROOT_WORLD_MIN_Z + static_cast<float>(gridZ) * tileExtent;

		uint32_t paddedW = TERRAIN_MAP_DIM + 2;
		uint32_t paddedH = TERRAIN_MAP_DIM + 2;
		size_t   totalPadded = static_cast<size_t>(paddedW) * paddedH;

		std::vector<float> paddedHeights(totalPadded);
		std::vector<float> biomePatch(totalPadded);
		std::vector<float> maskPatch(totalPadded);

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

		for (uint32_t z = 0; z < TERRAIN_MAP_DIM; ++z) {
			size_t rowIdx = static_cast<size_t>(z + 1) * paddedW;
			size_t prevRow = static_cast<size_t>(z) * paddedW;
			size_t nextRow = static_cast<size_t>(z + 2) * paddedW;

			for (uint32_t x = 0; x < TERRAIN_MAP_DIM; ++x) {
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

				float biomeVal = std::clamp(biomePatch[rowIdx + colIdx], 0.0f, 1.0f);
				float maskVal = maskPatch[rowIdx + colIdx];

				glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));

				size_t idx = static_cast<size_t>(z) * TERRAIN_MAP_DIM + x;

				data.heightMap[idx] = glm::vec4(h, normal.x, normal.y, normal.z);
				data.minMaxMap[idx] = glm::vec4(minH, maxH, minH, maxH);
				data.biomeMap[idx] = glm::vec4(biomeVal, maxH - minH, 1.0f, maskVal);
				data.visibilityMap[idx] = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
			}
		}

		return data;
	}

	TerrainClipmap::~TerrainClipmap() {
		Cleanup();
	}

	void TerrainClipmap::BuildQuadtree() {
		quadtreeNodes.resize(21845);
		for (uint32_t lod = 0; lod < numLODs; ++lod) {
			uint32_t dim = 128 >> lod;
			float    tileExtent = static_cast<float>(TERRAIN_MAP_DIM) * (baseTexelSize * static_cast<float>(1 << lod));

			for (uint32_t z = 0; z < dim; ++z) {
				for (uint32_t x = 0; x < dim; ++x) {
					size_t idx = GetNodeIndex(lod, x, z);
					auto&  node = quadtreeNodes[idx];
					node.lod = lod;
					node.gridX = x;
					node.gridZ = z;
					node.minWorld = glm::vec2(ROOT_WORLD_MIN_X, ROOT_WORLD_MIN_Z) + glm::vec2(x, z) * tileExtent;
					node.maxWorld = node.minWorld + glm::vec2(tileExtent);
					node.tileSlot = -1;
					node.isLoaded = false;
				}
			}
		}

		freeTileSlots.resize(numTileSlots);
		for (uint32_t i = 0; i < numTileSlots; ++i) {
			freeTileSlots[i] = i;
		}

		indirectionCpuMips.resize(numLODs);
		for (uint32_t l = 0; l < numLODs; ++l) {
			uint32_t dim = 128 >> l;
			indirectionCpuMips[l].assign(static_cast<size_t>(dim) * dim, INVALID_TILE_INDEX_FLOAT);
		}
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
		numLODs = lods;
		numTileSlots = DEFAULT_SLOT_CAPACITY;
		baseTexelSize = baseTexel;

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

		BuildQuadtree();
		if (device) {
			CreateTextureArrays();
			CreateIndirectionMapImage();
			CreateSampler();
		}
	}

	void TerrainClipmap::UpdateCameraPosition(const glm::vec3& cameraPos, AsyncTerrainUploader& uploader, vk::Queue queue) {
		glm::vec2 cam2D(cameraPos.x, cameraPos.z);

		static const float lodRadii[8] = {
			1024.0f,  // LOD 0
			2048.0f,  // LOD 1
			4096.0f,  // LOD 2
			8192.0f,  // LOD 3
			16384.0f, // LOD 4
			32768.0f, // LOD 5
			65536.0f, // LOD 6
			1e9f      // LOD 7 (Root)
		};

		std::vector<bool> desiredActive(quadtreeNodes.size(), false);

		for (uint32_t lod = 0; lod < numLODs; ++lod) {
			uint32_t dim = 128 >> lod;
			float radius = lodRadii[lod];

			for (uint32_t z = 0; z < dim; ++z) {
				for (uint32_t x = 0; x < dim; ++x) {
					size_t idx = GetNodeIndex(lod, x, z);
					const auto& node = quadtreeNodes[idx];

					glm::vec2 center = (node.minWorld + node.maxWorld) * 0.5f;
					float dist = glm::length(center - cam2D);

					if (dist <= radius) {
						desiredActive[idx] = true;
					}
				}
			}
		}

		for (size_t idx = 0; idx < quadtreeNodes.size(); ++idx) {
			auto& node = quadtreeNodes[idx];
			if (node.isLoaded && !desiredActive[idx]) {
				if (node.tileSlot >= 0) {
					freeTileSlots.push_back(static_cast<uint32_t>(node.tileSlot));
					node.tileSlot = -1;
				}
				node.isLoaded = false;
			}
		}

		for (int lod = static_cast<int>(numLODs) - 1; lod >= 0; --lod) {
			uint32_t dim = 128 >> lod;
			for (uint32_t z = 0; z < dim; ++z) {
				for (uint32_t x = 0; x < dim; ++x) {
					size_t idx = GetNodeIndex(static_cast<uint32_t>(lod), x, z);
					auto& node = quadtreeNodes[idx];

					if (desiredActive[idx] && !node.isLoaded && !freeTileSlots.empty()) {
						uint32_t slot = freeTileSlots.back();
						freeTileSlots.pop_back();

						node.tileSlot = static_cast<int>(slot);
						node.isLoaded = true;

						TerrainLevelData tileData = GenerateTileData(static_cast<uint32_t>(lod), x, z);
						uploader.UploadLevelAsync(slot, tileData.heightMap, image, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
						uploader.UploadLevelAsync(slot, tileData.minMaxMap, minmaxImage, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
						uploader.UploadLevelAsync(slot, tileData.biomeMap, biomeImage, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
						uploader.UploadLevelAsync(slot, tileData.visibilityMap, visibilityImage, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
					}
				}
			}
		}

		for (uint32_t l = 0; l < numLODs; ++l) {
			uint32_t dim = 128 >> l;
			bool changed = false;

			for (uint32_t z = 0; z < dim; ++z) {
				for (uint32_t x = 0; x < dim; ++x) {
					size_t idx = GetNodeIndex(l, x, z);
					const auto& node = quadtreeNodes[idx];

					float targetVal = (node.isLoaded && node.tileSlot >= 0)
						? static_cast<float>(node.tileSlot)
						: INVALID_TILE_INDEX_FLOAT;

					size_t mipIdx = static_cast<size_t>(z) * dim + x;
					if (indirectionCpuMips[l][mipIdx] != targetVal) {
						indirectionCpuMips[l][mipIdx] = targetVal;
						changed = true;
					}
				}
			}

			if (changed && indirectionImage) {
				uploader.UploadMipAsync(l, indirectionCpuMips[l], indirectionImage, dim, dim, queue);
			}
		}
	}

	void TerrainClipmap::Cleanup() {
		if (device) {
			if (sampler) {
				device.destroySampler(sampler);
				sampler = nullptr;
			}
			auto destroyImageRes = [&](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
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
			destroyImageRes(image, imageView, allocation);
			destroyImageRes(minmaxImage, minmaxImageView, minmaxAllocation);
			destroyImageRes(biomeImage, biomeImageView, biomeAllocation);
			destroyImageRes(visibilityImage, visibilityImageView, visibilityAllocation);
			destroyImageRes(indirectionImage, indirectionImageView, indirectionAllocation);
		}
	}

	void TerrainClipmap::CreateTextureArrays() {
		auto createArrayImage = [&](vk::Image& img, vk::ImageView& view, VmaAllocation& alloc) {
			VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent = VkExtent3D{TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 1};
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = numTileSlots;
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
			viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, numTileSlots));
			view = device.createImageView(viewInfo);
		};

		createArrayImage(image, imageView, allocation);
		createArrayImage(minmaxImage, minmaxImageView, minmaxAllocation);
		createArrayImage(biomeImage, biomeImageView, biomeAllocation);
		createArrayImage(visibilityImage, visibilityImageView, visibilityAllocation);
	}

	void TerrainClipmap::CreateIndirectionMapImage() {
		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = VkExtent3D{INDIRECTION_MAP_DIM, INDIRECTION_MAP_DIM, 1};
		imageInfo.mipLevels = numLODs;
		imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R32_SFLOAT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		VkImage vkImg = VK_NULL_HANDLE;
		if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &indirectionAllocation, nullptr) != VK_SUCCESS) {
			spdlog::error("Failed to create TerrainClipmap indirection map image!");
			return;
		}
		indirectionImage = vkImg;

		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.setImage(indirectionImage);
		viewInfo.setViewType(vk::ImageViewType::e2D);
		viewInfo.setFormat(vk::Format::eR32Sfloat);
		viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, numLODs, 0, 1));
		indirectionImageView = device.createImageView(viewInfo);
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
		return GenerateTileData(levelIndex, 0, 0).heightMap;
	}

	TerrainClipmap::TerrainLevelData TerrainClipmap::GenerateLevelData(uint32_t levelIndex) const {
		return GenerateTileData(levelIndex, 0, 0);
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

		TerrainClipmap dummy;
		dummy.baseTexelSize = baseTexelSize;
		return dummy.GenerateTileData(levelIndex, 0, 0).heightMap;
	}

} // namespace brassica
