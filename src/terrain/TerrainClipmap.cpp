#include "terrain/TerrainClipmap.hpp"

#include <algorithm>
#include <cmath>

#include "spdlog/spdlog.h"
#include <FastNoise/FastNoise.h>

#include "terrain/AsyncTerrainUploader.hpp"
#include <glm/gtc/matrix_transform.hpp>

namespace brassica {

	struct TerrainNoiseGenerators {
		FastNoise::SmartNode<FastNoise::DomainScale> baseScale;
		FastNoise::SmartNode<FastNoise::DomainScale> detailScale;
		FastNoise::SmartNode<FastNoise::DomainScale> maskScale;
		FastNoise::SmartNode<FastNoise::DomainScale> biomeScale;
		FastNoise::SmartNode<FastNoise::DomainScale> land;

		TerrainNoiseGenerators() {
			land = FastNoise::New<FastNoise::DomainScale>();
			auto landGen = FastNoise::NewFromEncodedNodeTree(
				"KQkOCRYCFwkZCQYAAEAcRgwCEwkQ@B+kQwAQ@BkL@BekQS/wAABAOamRNCBAMK16M+Cv8BAAwK/wQABAL/"
				"BAAEAw@CTAABIwhsAAHpEBA=="
			);
			land->SetSource(landGen);

			auto simplex = FastNoise::New<FastNoise::Simplex>();

			auto baseFbm = FastNoise::New<FastNoise::FractalFBm>();
			baseFbm->SetSource(simplex);
			baseFbm->SetOctaveCount(2);
			baseFbm->SetLacunarity(2.0f);
			baseFbm->SetGain(0.5f);

			baseScale = FastNoise::New<FastNoise::DomainScale>();
			baseScale->SetSource(baseFbm);
			baseScale->SetScaling(0.01f);

			auto detailFbm = FastNoise::New<FastNoise::FractalFBm>();
			detailFbm->SetSource(simplex);
			detailFbm->SetOctaveCount(6);
			detailFbm->SetLacunarity(2.0f);
			detailFbm->SetGain(0.5f);

			detailScale = FastNoise::New<FastNoise::DomainScale>();
			detailScale->SetSource(detailFbm);
			detailScale->SetScaling(0.4f);

			maskScale = FastNoise::New<FastNoise::DomainScale>();
			maskScale->SetSource(simplex);
			maskScale->SetScaling(0.0008f);

			// Domain Warped Worley Noise for Biomes
			auto cellular = FastNoise::New<FastNoise::CellularDistance>();
			cellular->SetDistanceFunction(FastNoise::DistanceFunction::Euclidean);
			cellular->SetReturnType(FastNoise::CellularDistance::ReturnType::Index0);

			auto warp = FastNoise::New<FastNoise::DomainWarpGradient>();
			warp->SetWarpAmplitude(50.0f);
			warp->SetSource(cellular);

			biomeScale = FastNoise::New<FastNoise::DomainScale>();
			biomeScale->SetSource(warp);
			biomeScale->SetScaling(0.0004f);
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
			float maskVal = gens.maskScale->GenSingle2D(x, z, seed);
			float baseVal = gens.baseScale->GenSingle2D(x, z, seed);
			float detailVal = gens.detailScale->GenSingle2D(x, z, seed);
			float biomeVal = gens.biomeScale->GenSingle2D(x, z, seed);

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

	static std::vector<glm::vec4> GenerateTerrainRegion(
		const ClipmapLevelInfo& info,
		uint32_t                startX,
		uint32_t                startZ,
		uint32_t                width,
		uint32_t                height,
		int                     deltaX = 0,
		int                     deltaZ = 0
	) {
		std::vector<glm::vec4> stripData(width * height);
		float                  texelSize = info.texelSize;
		float                  halfExtent = 0.5f * static_cast<float>(TERRAIN_MAP_DIM) * texelSize;

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

		std::vector<float> basePatch(totalPadded);
		std::vector<float> detailPatch(totalPadded);
		std::vector<float> maskPatch(totalPadded);
		std::vector<float> biomePatch(totalPadded);
		std::vector<float> paddedHeights(totalPadded);

		auto&         gens = GetGenerators();
		constexpr int seed = 1337;

		float gridStartX = minWorldX - texelSize;
		float gridStartZ = minWorldZ - texelSize;

		// gens.baseScale->GenUniformGrid2D(basePatch.data(), gridStartX, gridStartZ, paddedW, paddedH, texelSize,
		// texelSize, seed); gens.detailScale->GenUniformGrid2D(detailPatch.data(), gridStartX, gridStartZ, paddedW,
		// paddedH, texelSize, texelSize, seed); gens.maskScale->GenUniformGrid2D(maskPatch.data(), gridStartX,
		// gridStartZ, paddedW, paddedH, texelSize, texelSize, seed);
		// gens.biomeScale->GenUniformGrid2D(biomePatch.data(), gridStartX, gridStartZ, paddedW, paddedH, texelSize,
		// texelSize, seed);

		// for (size_t i = 0; i < totalPadded; ++i) {
		// 	paddedHeights[i] = EvalHeightFromComponents(basePatch[i], detailPatch[i], maskPatch[i], biomePatch[i]);
		// }

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

		float eps = std::max(0.25f, texelSize);

		for (uint32_t z = 0; z < height; ++z) {
			size_t rowIdx = static_cast<size_t>(z + 1) * paddedW;
			size_t prevRow = static_cast<size_t>(z) * paddedW;
			size_t nextRow = static_cast<size_t>(z + 2) * paddedW;

			// Apply toroidal wrap to the Z-axis if this is a full-height deltaX strip
			uint32_t destZ = (height == TERRAIN_MAP_DIM) ? ((z + info.gridOffset.y) % TERRAIN_MAP_DIM) : z;

			for (uint32_t x = 0; x < width; ++x) {
				size_t colIdx = x + 1;
				float  h = paddedHeights[rowIdx + colIdx];
				float  hL = paddedHeights[rowIdx + x];
				float  hR = paddedHeights[rowIdx + x + 2];
				float  hD = paddedHeights[prevRow + colIdx];
				float  hU = paddedHeights[nextRow + colIdx];

				glm::vec3 normal = glm::normalize(glm::vec3(hL - hR, 2.0f * eps, hD - hU));

				// Apply toroidal wrap to the X-axis if this is a full-width deltaZ strip
				uint32_t destX = (width == TERRAIN_MAP_DIM) ? ((x + info.gridOffset.x) % TERRAIN_MAP_DIM) : x;

				stripData[destZ * width + destX] = glm::vec4(h, normal.x, normal.y, normal.z);
			}
		}

		return stripData;
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

		CreateTextureArray();
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
				auto mapData = GenerateLevelMap(l);
				uploader.UploadLevelAsync(l, mapData, image, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, queue);
				continue;
			}

			info.centerWorldPos = newCenter;

			std::vector<glm::vec4>           updateBuffer;
			std::vector<vk::BufferImageCopy> copyRegions;

			if (deltaX != 0) {
				uint32_t stripWidth = std::abs(deltaX);
				int      startDstX = (deltaX > 0) ? info.gridOffset.x
												  : ((info.gridOffset.x + deltaX + static_cast<int>(TERRAIN_MAP_DIM)) %
												     static_cast<int>(TERRAIN_MAP_DIM));

				std::vector<glm::vec4> stripData =
					GenerateTerrainRegion(info, 0, 0, stripWidth, TERRAIN_MAP_DIM, deltaX, 0);

				size_t baseOffset = updateBuffer.size() * sizeof(glm::vec4);
				updateBuffer.insert(updateBuffer.end(), stripData.begin(), stripData.end());

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

				std::vector<glm::vec4> stripData =
					GenerateTerrainRegion(info, 0, 0, TERRAIN_MAP_DIM, stripHeight, 0, deltaZ);

				size_t baseOffset = updateBuffer.size() * sizeof(glm::vec4);
				updateBuffer.insert(updateBuffer.end(), stripData.begin(), stripData.end());

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
		viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, numLODs));
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

	std::vector<glm::vec4> TerrainClipmap::GenerateLevelMap(uint32_t levelIndex) const {
		return GenerateTerrainRegion(levelInfos[levelIndex], 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0);
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

		return GenerateTerrainRegion(info, 0, 0, TERRAIN_MAP_DIM, TERRAIN_MAP_DIM, 0, 0);
	}

} // namespace brassica
