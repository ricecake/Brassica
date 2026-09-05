#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	constexpr uint32_t TERRAIN_MAP_DIM = 1024;
	constexpr uint32_t DEFAULT_CLIPMAP_LODS = 4;

	struct ClipmapLevelInfo {
		uint32_t level{0};
		float    baseTexelSize{0.5f};
		float    texelSize{0.5f}; // texelSize = baseTexelSize * 2^level
		float    worldExtent{512.0f}; // 1024 * texelSize
		glm::vec2 centerWorldPos{0.0f};
	};

	class TerrainClipmap {
	public:
		TerrainClipmap() = default;
		~TerrainClipmap();

		void Init(vk::Device device, VmaAllocator allocator, uint32_t numLODs = DEFAULT_CLIPMAP_LODS, float baseTexelSize = 0.5f);
		void Cleanup();

		// CPU Terrain Generator function for a 1024x1024 grid at a specific clipmap level
		static std::vector<glm::vec4> GenerateSineWaveMap(
			uint32_t levelIndex,
			float baseTexelSize,
			const glm::vec2& centerWorldPos = glm::vec2(0.0f),
			float time = 0.0f
		);

		vk::Image GetImage() const { return image; }
		vk::ImageView GetImageView() const { return imageView; }
		vk::Sampler GetSampler() const { return sampler; }
		uint32_t GetNumLODs() const { return numLODs; }
		float GetBaseTexelSize() const { return baseTexelSize; }
		const ClipmapLevelInfo& GetLevelInfo(uint32_t lod) const { return levelInfos[lod]; }

	private:
		vk::Device    device{nullptr};
		VmaAllocator  allocator{VK_NULL_HANDLE};
		uint32_t      numLODs{DEFAULT_CLIPMAP_LODS};
		float         baseTexelSize{0.5f};

		vk::Image     image{nullptr};
		vk::ImageView imageView{nullptr};
		vk::Sampler   sampler{nullptr};
		VmaAllocation allocation{VK_NULL_HANDLE};

		std::vector<ClipmapLevelInfo> levelInfos;

		void CreateTextureArray();
		void CreateSampler();
	};

} // namespace brassica
