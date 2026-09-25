#pragma once

#include <cstdint>
#include <vector>

#include "VulkanCompat.hpp"

#ifndef BRASSICA_HAS_VULKAN
	#if __has_include(<vulkan/vulkan.hpp>) || __has_include("vulkan/vulkan.hpp")
		#define BRASSICA_HAS_VULKAN 1
	#else
		#define BRASSICA_HAS_VULKAN 0
	#endif
#endif

#if BRASSICA_HAS_VULKAN
	#include "vk_mem_alloc.h"
#else
	using VmaAllocator = void*;
	using VmaAllocation = void*;
#endif

#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "terrain/ITerrainClipmap.hpp"

namespace brassica {

	constexpr uint32_t TERRAIN_MAP_DIM = constants::Class::Terrain::MapDim;
	constexpr uint32_t DEFAULT_CLIPMAP_LODS = constants::Class::Terrain::DefaultMaxLODs;

	inline float GetLODScale(float lod) {
		if (lod <= 3.0f) {
			return std::pow(2.0f, lod);
		} else {
			return 8.0f * std::pow(2.25f, lod - 3.0f);
		}
	}

	struct ClipmapLevelInfo {
		uint32_t   level{0};
		float      baseTexelSize{0.5f};
		float      texelSize{0.5f};     //  texelSize = baseTexelSize * GetLODScale(level)
		float      worldExtent{512.0f}; // 1024 * texelSize
		glm::vec2  centerWorldPos{0.0f};
		glm::ivec2 gridOffset{0};                           // Toroidal grid cell offset in texels
		glm::ivec2 delta{TERRAIN_MAP_DIM, TERRAIN_MAP_DIM}; // Texel shift since last update
	};

	class AsyncTerrainUploader;

	class TerrainClipmap: public ITerrainClipmap {
	public:
		TerrainClipmap() = default;
		~TerrainClipmap() override;

		void Initialize() override { m_initialized = true; }

		void Shutdown() override {
			Cleanup();
			m_initialized = false;
		}

		State GetState() const override { return State{numLODs, baseTexelSize}; }

		void SetState(const State& state) override {
			numLODs = state.numLODs;
			baseTexelSize = state.baseTexelSize;
		}

		void Init(
			vk::Device       device,
			VmaAllocator     allocator,
			uint32_t         numLODs = DEFAULT_CLIPMAP_LODS,
			float            baseTexelSize = constants::Class::Terrain::BaseTexelSize,
			float            maxDistance = 0.0f,
			const glm::vec3& initialCameraPos = glm::vec3(0.0f)
		);
		void Cleanup();

		void UpdateCameraPosition(const glm::vec3& cameraPos);

		void Regenerate() override { m_forceRegenerate = true; }

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

		static std::vector<glm::vec4> GenerateSineWaveMap(
			uint32_t         levelIndex,
			float            baseTexelSize,
			const glm::vec2& centerWorldPos = glm::vec2(0.0f),
			float            time = 0.0f
		);

		vk::Image GetImage() const {
#if BRASSICA_HAS_VULKAN
			return image;
#else
			return {};
#endif
		}

		vk::ImageView GetImageView() const {
#if BRASSICA_HAS_VULKAN
			return imageView;
#else
			return {};
#endif
		}

		vk::Image GetMinMaxImage() const {
#if BRASSICA_HAS_VULKAN
			return minmaxImage;
#else
			return {};
#endif
		}

		vk::ImageView GetMinMaxImageView() const {
#if BRASSICA_HAS_VULKAN
			return minmaxImageView;
#else
			return {};
#endif
		}

		vk::Image GetBiomeImage() const {
#if BRASSICA_HAS_VULKAN
			return biomeImage;
#else
			return {};
#endif
		}

		vk::ImageView GetBiomeImageView() const {
#if BRASSICA_HAS_VULKAN
			return biomeImageView;
#else
			return {};
#endif
		}

		vk::Image GetVisibilityImage() const {
#if BRASSICA_HAS_VULKAN
			return visibilityImage;
#else
			return {};
#endif
		}

		vk::ImageView GetVisibilityImageView() const {
#if BRASSICA_HAS_VULKAN
			return visibilityImageView;
#else
			return {};
#endif
		}

		vk::Sampler GetSampler() const {
#if BRASSICA_HAS_VULKAN
			return sampler;
#else
			return {};
#endif
		}

		uint32_t GetNumLODs() const { return numLODs; }

		float GetBaseTexelSize() const { return baseTexelSize; }

		const ClipmapLevelInfo& GetLevelInfo(uint32_t lod) const { return levelInfos[lod]; }

	private:
		vk::Device   device{};
		VmaAllocator allocator{nullptr};
		uint32_t     numLODs{DEFAULT_CLIPMAP_LODS};
		float        baseTexelSize{0.5f};

#if BRASSICA_HAS_VULKAN
		vk::Image     image{nullptr};
		vk::ImageView imageView{nullptr};
		VmaAllocation allocation{nullptr};

		vk::Image     minmaxImage{nullptr};
		vk::ImageView minmaxImageView{nullptr};
		VmaAllocation minmaxAllocation{nullptr};

		vk::Image     biomeImage{nullptr};
		vk::ImageView biomeImageView{nullptr};
		VmaAllocation biomeAllocation{nullptr};

		vk::Image     visibilityImage{nullptr};
		vk::ImageView visibilityImageView{nullptr};
		VmaAllocation visibilityAllocation{nullptr};

		vk::Sampler sampler{nullptr};
#endif

		std::vector<ClipmapLevelInfo> levelInfos;
		bool                          m_forceRegenerate{false};

		void CreateTextureArrays();
		void CreateSampler();
	};

	inline graph::ResourceDesc TerrainClipmapDesc(std::uint32_t numLODs) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = TERRAIN_MAP_DIM,
			.height = TERRAIN_MAP_DIM,
			.layers = numLODs,
			.formatCode = static_cast<std::uint32_t>(
#if BRASSICA_HAS_VULKAN
				vk::Format::eR32G32B32A32Sfloat
#else
				0
#endif
			),
			.usageMask = static_cast<std::uint32_t>(
#if BRASSICA_HAS_VULKAN
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
#else
				0
#endif
			),
		};
	}

	inline graph::ResourceDesc TerrainMinMaxDesc(std::uint32_t numLODs) {
		return TerrainClipmapDesc(numLODs);
	}

	inline graph::ResourceDesc TerrainBiomeDesc(std::uint32_t numLODs) {
		return TerrainClipmapDesc(numLODs);
	}

	inline graph::ResourceDesc TerrainTileVisibilityDesc(std::uint32_t numLODs) {
		return TerrainClipmapDesc(numLODs);
	}

} // namespace brassica
