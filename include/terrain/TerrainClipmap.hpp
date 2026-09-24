#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "terrain/ITerrainClipmap.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	constexpr uint32_t TERRAIN_MAP_DIM = constants::Class::Terrain::MapDim;
	constexpr uint32_t DEFAULT_CLIPMAP_LODS = constants::Class::Terrain::DefaultMaxLODs;

	inline float GetLODScale(float lod) {
		return std::pow(2.0f, lod);
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

		State GetState() const override { return State{numLODs, windowLODs, baseTexelSize}; }

		void SetState(const State& state) override {
			numLODs = state.numLODs;
			windowLODs = state.windowLODs;
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

		vk::Image GetImage() const { return image; }

		vk::ImageView GetImageView() const { return imageView; }

		vk::Image GetMinMaxImage() const { return minmaxImage; }

		vk::ImageView GetMinMaxImageView() const { return minmaxImageView; }

		vk::Image GetBiomeImage() const { return biomeImage; }

		vk::ImageView GetBiomeImageView() const { return biomeImageView; }

		vk::Image GetVisibilityImage() const { return visibilityImage; }

		vk::ImageView GetVisibilityImageView() const { return visibilityImageView; }

		vk::Sampler GetSampler() const { return sampler; }

		uint32_t GetNumLODs() const { return numLODs; }

		uint32_t GetWindowLODs() const { return windowLODs; }

		uint32_t GetBaseLOD() const { return baseLOD; }

		float GetBaseTexelSize() const { return baseTexelSize; }

		static uint32_t CalculateBaseLOD(float height, uint32_t totalLODs, uint32_t activeWindow) {
			return brassica::CalculateBaseLOD(height, totalLODs, activeWindow);
		}

		const ClipmapLevelInfo& GetLevelInfo(uint32_t lod) const { return levelInfos[lod]; }

	private:
		vk::Device   device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};
		uint32_t     numLODs{DEFAULT_CLIPMAP_LODS};
		uint32_t     windowLODs{constants::Class::Terrain::DefaultWindowLODs};
		uint32_t     baseLOD{0};
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

		vk::Sampler sampler{nullptr};

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
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage
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
