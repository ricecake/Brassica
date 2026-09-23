#include "terrain/TerrainClipmap.hpp"

#include <algorithm>
#include <cmath>

#include "spdlog/spdlog.h"
#include <FastNoise/FastNoise.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
			while (derivedLODs < 12 &&
			       (static_cast<float>(TERRAIN_MAP_DIM) * GetLODScale(static_cast<float>(derivedLODs - 1)) *
			        baseTexelSize * 0.5f) < maxDist) {
				derivedLODs++;
			}
			numLODs = std::clamp(derivedLODs, 1u, 12u);
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

		if (allocator != VK_NULL_HANDLE) {
			VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = TERRAIN_MAP_DIM * TERRAIN_MAP_DIM * sizeof(glm::vec4);
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer vkBuf = VK_NULL_HANDLE;
			VmaAllocationInfo allocResultInfo{};
			if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &vkBuf, &m_readbackAllocation, &allocResultInfo) == VK_SUCCESS) {
				m_readbackBuffer = vkBuf;
				m_readbackMappedPtr = allocResultInfo.pMappedData;
				m_cpuPhysicsLODHeightmap.resize(TERRAIN_MAP_DIM * TERRAIN_MAP_DIM, glm::vec4(0.0f));
			} else {
				spdlog::error("Failed to create TerrainClipmap readback buffer!");
			}
		}
	}

	void TerrainClipmap::UpdateCameraPosition(const glm::vec3& cameraPos) {
		for (uint32_t l = 0; l < numLODs; ++l) {
			auto& info = levelInfos[l];
			float texelSize = info.texelSize;

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

			if (m_readbackBuffer && m_readbackAllocation && allocator != VK_NULL_HANDLE) {
				vmaDestroyBuffer(allocator, m_readbackBuffer, m_readbackAllocation);
				m_readbackBuffer = nullptr;
				m_readbackAllocation = VK_NULL_HANDLE;
				m_readbackMappedPtr = nullptr;
			}
		}
	}

	void TerrainClipmap::RecordReadbackCommand(vk::CommandBuffer cmd) {
		if (!image || !m_readbackBuffer) {
			return;
		}

		vk::ImageMemoryBarrier2 barrierToSrc{};
		barrierToSrc.setSrcStageMask(vk::PipelineStageFlagBits2::eComputeShader);
		barrierToSrc.setSrcAccessMask(vk::AccessFlagBits2::eShaderStorageWrite);
		barrierToSrc.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrierToSrc.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
		barrierToSrc.setOldLayout(vk::ImageLayout::eGeneral);
		barrierToSrc.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrierToSrc.setImage(image);
		barrierToSrc.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		vk::DependencyInfo depToSrc{};
		depToSrc.setImageMemoryBarriers(barrierToSrc);
		cmd.pipelineBarrier2(depToSrc);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setBufferOffset(0);
		copyRegion.setBufferRowLength(TERRAIN_MAP_DIM);
		copyRegion.setBufferImageHeight(TERRAIN_MAP_DIM);
		copyRegion.setSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1));
		copyRegion.setImageOffset(vk::Offset3D{0, 0, 0});
		copyRegion.setImageExtent(vk::Extent3D{TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 1});

		cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, m_readbackBuffer, copyRegion);

		vk::ImageMemoryBarrier2 barrierBack{};
		barrierBack.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrierBack.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead);
		barrierBack.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
		barrierBack.setDstAccessMask(
			vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
			vk::AccessFlagBits2::eShaderSampledRead
		);
		barrierBack.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrierBack.setNewLayout(vk::ImageLayout::eGeneral);
		barrierBack.setImage(image);
		barrierBack.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		vk::DependencyInfo depBack{};
		depBack.setImageMemoryBarriers(barrierBack);
		cmd.pipelineBarrier2(depBack);

		if (!levelInfos.empty()) {
			m_cpuPhysicsLODInfo = levelInfos[0];
		}
	}

	void TerrainClipmap::SyncCPUHeightmap() {
		if (!m_readbackMappedPtr || m_cpuPhysicsLODHeightmap.empty()) {
			return;
		}
		if (allocator != VK_NULL_HANDLE && m_readbackAllocation != VK_NULL_HANDLE) {
			vmaInvalidateAllocation(allocator, m_readbackAllocation, 0, VK_WHOLE_SIZE);
		}
		std::memcpy(
			m_cpuPhysicsLODHeightmap.data(),
			m_readbackMappedPtr,
			TERRAIN_MAP_DIM * TERRAIN_MAP_DIM * sizeof(glm::vec4)
		);
		m_hasValidCpuReadback = true;
	}

	bool TerrainClipmap::GetHeightAtWorldPos(float worldX, float worldZ, float& outHeight) const {
		if (!m_hasValidCpuReadback || m_cpuPhysicsLODHeightmap.empty()) {
			return false;
		}

		const auto& info = m_cpuPhysicsLODInfo;
		float       texelSize = info.texelSize > 0.0f ? info.texelSize : 0.5f;
		float       halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

		float minWorldX = info.centerWorldPos.x - halfExtent;
		float minWorldZ = info.centerWorldPos.y - halfExtent;

		float localX = (worldX - minWorldX) / texelSize;
		float localZ = (worldZ - minWorldZ) / texelSize;

		if (localX < 0.0f || localX >= static_cast<float>(TERRAIN_MAP_DIM - 1) || localZ < 0.0f ||
		    localZ >= static_cast<float>(TERRAIN_MAP_DIM - 1)) {
			return false;
		}

		int x0 = static_cast<int>(std::floor(localX));
		int z0 = static_cast<int>(std::floor(localZ));
		int x1 = std::min(x0 + 1, static_cast<int>(TERRAIN_MAP_DIM - 1));
		int z1 = std::min(z0 + 1, static_cast<int>(TERRAIN_MAP_DIM - 1));

		float fx = localX - static_cast<float>(x0);
		float fz = localZ - static_cast<float>(z0);

		auto getTexelHeight = [&](int col, int row) -> float {
			int destX = (col + info.gridOffset.x) % static_cast<int>(TERRAIN_MAP_DIM);
			if (destX < 0)
				destX += static_cast<int>(TERRAIN_MAP_DIM);
			int destZ = (row + info.gridOffset.y) % static_cast<int>(TERRAIN_MAP_DIM);
			if (destZ < 0)
				destZ += static_cast<int>(TERRAIN_MAP_DIM);
			size_t idx = static_cast<size_t>(destZ) * TERRAIN_MAP_DIM + static_cast<size_t>(destX);
			return m_cpuPhysicsLODHeightmap[idx].r;
		};

		float h00 = getTexelHeight(x0, z0);
		float h10 = getTexelHeight(x1, z0);
		float h01 = getTexelHeight(x0, z1);
		float h11 = getTexelHeight(x1, z1);

		outHeight = std::lerp(std::lerp(h00, h10, fx), std::lerp(h01, h11, fx), fz);
		return true;
	}

	float TerrainClipmap::SampleHeight(float worldX, float worldZ) const {
		float h = 0.0f;
		if (GetHeightAtWorldPos(worldX, worldZ, h)) {
			return h;
		}
		float texelSize = levelInfos.empty() ? 0.5f : levelInfos[0].texelSize;
		return TerrainClipmap::SampleTerrain(worldX, worldZ, texelSize).r;
	}

	bool TerrainClipmap::ExportTerrainMapPNG(
		const std::string& filepath,
		glm::vec2          centerWorldPos,
		float              chunkExtent,
		uint32_t           resolution
	) const {
		if (resolution == 0 || chunkExtent <= 0.0f) {
			spdlog::error("Invalid resolution or chunk extent for terrain map PNG export.");
			return false;
		}

		std::vector<uint8_t> pixels(static_cast<size_t>(resolution) * resolution * 4);
		float                texelSize = chunkExtent / static_cast<float>(resolution);
		float                halfExtent = 0.5f * chunkExtent;

		auto sampleColor = [](float h, const glm::vec3& N) -> glm::u8vec3 {
			glm::vec3 col(0.0f);
			if (h < 0.0f) {
				float t = std::clamp((h + 50.0f) / 50.0f, 0.0f, 1.0f);
				col = glm::mix(glm::vec3(15.0f, 23.0f, 42.0f), glm::vec3(14.0f, 165.0f, 233.0f), t) / 255.0f;
			} else if (h < 4.0f) {
				float t = h / 4.0f;
				col = glm::mix(glm::vec3(254.0f, 240.0f, 138.0f), glm::vec3(234.0f, 179.0f, 8.0f), t) / 255.0f;
			} else if (h < 40.0f) {
				float t = (h - 4.0f) / 36.0f;
				col = glm::mix(glm::vec3(34.0f, 197.0f, 94.0f), glm::vec3(21.0f, 128.0f, 61.0f), t) / 255.0f;
			} else if (h < 120.0f) {
				float t = (h - 40.0f) / 80.0f;
				col = glm::mix(glm::vec3(22.0f, 101.0f, 52.0f), glm::vec3(133.0f, 77.0f, 14.0f), t) / 255.0f;
			} else if (h < 220.0f) {
				float t = (h - 120.0f) / 100.0f;
				col = glm::mix(glm::vec3(100.0f, 116.0f, 139.0f), glm::vec3(51.0f, 65.0f, 85.0f), t) / 255.0f;
			} else {
				float t = std::clamp((h - 220.0f) / 100.0f, 0.0f, 1.0f);
				col = glm::mix(glm::vec3(226.0f, 232.0f, 240.0f), glm::vec3(255.0f, 255.0f, 255.0f), t) / 255.0f;
			}

			glm::vec3 lightDir = glm::normalize(glm::vec3(-0.5f, 0.8f, -0.5f));
			float     hillshade = std::clamp(glm::dot(N, lightDir), 0.35f, 1.25f);
			col *= hillshade;

			col = glm::clamp(col * 255.0f, glm::vec3(0.0f), glm::vec3(255.0f));
			return glm::u8vec3(col.r, col.g, col.b);
		};

		for (uint32_t y = 0; y < resolution; ++y) {
			float worldZ = centerWorldPos.y - halfExtent + (static_cast<float>(y) + 0.5f) * texelSize;
			for (uint32_t x = 0; x < resolution; ++x) {
				float worldX = centerWorldPos.x - halfExtent + (static_cast<float>(x) + 0.5f) * texelSize;

				glm::vec4 sample = SampleTerrain(worldX, worldZ, texelSize);
				float     h = sample.r;
				glm::vec3 N(sample.g, sample.b, sample.a);

				glm::u8vec3 rgb = sampleColor(h, N);

				size_t idx = (static_cast<size_t>(y) * resolution + x) * 4;
				pixels[idx + 0] = rgb.r;
				pixels[idx + 1] = rgb.g;
				pixels[idx + 2] = rgb.b;
				pixels[idx + 3] = 255;
			}
		}

		int resInt = static_cast<int>(resolution);
		int success = stbi_write_png(filepath.c_str(), resInt, resInt, 4, pixels.data(), resInt * 4);
		if (success) {
			spdlog::info(
				"Exported terrain map PNG to '{}' ({}x{} @ {}m extent)",
				filepath,
				resolution,
				resolution,
				chunkExtent
			);
			return true;
		} else {
			spdlog::error("Failed to write terrain map PNG to '{}'", filepath);
			return false;
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
