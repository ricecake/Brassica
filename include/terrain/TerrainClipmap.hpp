#pragma once

#include <cstdint>
#include <vector>
#include <array>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	constexpr uint32_t TERRAIN_MAP_DIM = 1024; // 1024x1024 physical tile resolution
	constexpr uint32_t DEFAULT_CLIPMAP_LODS = 8;
	constexpr uint32_t DEFAULT_SLOT_CAPACITY = 64; // Capacity of 2D texture array layers
	constexpr uint32_t INDIRECTION_MAP_DIM = 128; // Mip 0 indirection map resolution (128x128)
	constexpr float    ROOT_WORLD_EXTENT = 65536.0f; // 128 * 512m
	constexpr float    ROOT_WORLD_MIN_X = -32768.0f;
	constexpr float    ROOT_WORLD_MIN_Z = -32768.0f;
	constexpr float    INVALID_TILE_INDEX_FLOAT = 65535.0f;

	struct ClipmapLevelInfo {
		uint32_t   level{0};
		float      baseTexelSize{0.5f};
		float      texelSize{0.5f};     // texelSize = baseTexelSize * 2^level
		float      worldExtent{512.0f}; // 1024 * texelSize
		glm::vec2  centerWorldPos{0.0f};
		glm::ivec2 gridOffset{0};
	};

	struct QuadtreeNode {
		uint32_t  lod{0};
		uint32_t  gridX{0};
		uint32_t  gridZ{0};
		glm::vec2 minWorld{0.0f};
		glm::vec2 maxWorld{0.0f};
		int       tileSlot{-1};
		bool      isLoaded{false};
	};

	class AsyncTerrainUploader;

	class TerrainClipmap {
	public:
		TerrainClipmap() = default;
		~TerrainClipmap();

		void Init(
			vk::Device       device,
			VmaAllocator     allocator,
			uint32_t         numLODs = DEFAULT_CLIPMAP_LODS,
			float            baseTexelSize = 0.5f,
			float            maxDistance = 32768.0f,
			const glm::vec3& initialCameraPos = glm::vec3(0.0f)
		);
		void Cleanup();

		void UpdateCameraPosition(const glm::vec3& cameraPos, AsyncTerrainUploader& uploader, vk::Queue queue);

		// Unified CPU Terrain Generator
		static glm::vec4 SampleTerrain(float worldX, float worldZ, float texelSize);

		struct TerrainLevelData {
			std::vector<glm::vec4> heightMap;
			std::vector<glm::vec4> minMaxMap;
			std::vector<glm::vec4> biomeMap;
			std::vector<glm::vec4> visibilityMap;
		};

		std::vector<glm::vec4> GenerateLevelMap(uint32_t levelIndex) const;
		TerrainLevelData       GenerateLevelData(uint32_t levelIndex) const;
		TerrainLevelData       GenerateTileData(uint32_t lod, uint32_t gridX, uint32_t gridZ) const;

		static std::vector<glm::vec4> GenerateSineWaveMap(
			uint32_t         levelIndex,
			float            baseTexelSize,
			const glm::vec2& centerWorldPos = glm::vec2(0.0f),
			float            time = 0.0f
		);

		vk::Image GetImage() const { return image; }

		vk::ImageView GetImageView() const { return imageView; }

		vk::Image GetMinMaxImage() const { return minmaxImage; }

		vk::ImageView GetMinMaxImageView() const { return minmaxImageView; }

		vk::Image GetBiomeImage() const { return biomeImage; }

		vk::ImageView GetBiomeImageView() const { return biomeImageView; }

		vk::Image GetVisibilityImage() const { return visibilityImage; }

		vk::ImageView GetVisibilityImageView() const { return visibilityImageView; }

		vk::Image GetIndirectionImage() const { return indirectionImage; }

		vk::ImageView GetIndirectionImageView() const { return indirectionImageView; }

		vk::Sampler GetSampler() const { return sampler; }

		uint32_t GetNumLODs() const { return numLODs; }

		uint32_t GetNumTileSlots() const { return numTileSlots; }

		float GetBaseTexelSize() const { return baseTexelSize; }

		const ClipmapLevelInfo& GetLevelInfo(uint32_t lod) const { return levelInfos[lod]; }

		const std::vector<QuadtreeNode>& GetQuadtreeNodes() const { return quadtreeNodes; }

	private:
		vk::Device   device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};
		uint32_t     numLODs{DEFAULT_CLIPMAP_LODS};
		uint32_t     numTileSlots{DEFAULT_SLOT_CAPACITY};
		float        baseTexelSize{0.5f};

		vk::Image     image{nullptr};
		vk::ImageView imageView{nullptr};
		VmaAllocation allocation{VK_NULL_HANDLE};

		vk::Image     minmaxImage{nullptr};
		vk::ImageView minmaxImageView{nullptr};
		VmaAllocation minmaxAllocation{VK_NULL_HANDLE};

		vk::Image     biomeImage{nullptr};
		vk::ImageView biomeImageView{nullptr};
		VmaAllocation biomeAllocation{VK_NULL_HANDLE};

		vk::Image     visibilityImage{nullptr};
		vk::ImageView visibilityImageView{nullptr};
		VmaAllocation visibilityAllocation{VK_NULL_HANDLE};

		vk::Image     indirectionImage{nullptr};
		vk::ImageView indirectionImageView{nullptr};
		VmaAllocation indirectionAllocation{VK_NULL_HANDLE};

		vk::Sampler sampler{nullptr};

		std::vector<ClipmapLevelInfo> levelInfos;
		std::vector<QuadtreeNode>     quadtreeNodes;
		std::vector<uint32_t>         freeTileSlots;
		std::vector<std::vector<float>> indirectionCpuMips;

		void BuildQuadtree();
		void CreateTextureArrays();
		void CreateIndirectionMapImage();
		void CreateSampler();
		size_t GetNodeIndex(uint32_t lod, uint32_t gridX, uint32_t gridZ) const;
	};

	inline graph::ResourceDesc TerrainClipmapDesc(std::uint32_t numLayers = DEFAULT_SLOT_CAPACITY) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = TERRAIN_MAP_DIM,
			.height = TERRAIN_MAP_DIM,
			.layers = numLayers,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	inline graph::ResourceDesc TerrainMinMaxDesc(std::uint32_t numLayers = DEFAULT_SLOT_CAPACITY) {
		return TerrainClipmapDesc(numLayers);
	}

	inline graph::ResourceDesc TerrainBiomeDesc(std::uint32_t numLayers = DEFAULT_SLOT_CAPACITY) {
		return TerrainClipmapDesc(numLayers);
	}

	inline graph::ResourceDesc TerrainTileVisibilityDesc(std::uint32_t numLayers = DEFAULT_SLOT_CAPACITY) {
		return TerrainClipmapDesc(numLayers);
	}

	inline graph::ResourceDesc TerrainIndirectionMapDesc(std::uint32_t numLODs = DEFAULT_CLIPMAP_LODS) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = INDIRECTION_MAP_DIM,
			.height = INDIRECTION_MAP_DIM,
			.layers = 1,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

} // namespace brassica
