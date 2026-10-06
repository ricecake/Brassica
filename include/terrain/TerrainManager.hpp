#pragma once

#include <cstdint>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "terrain/ITerrainManager.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	// CPU-side terrain config/state (numLODs/baseTexelSize, Regenerate()) plus the async GPU->CPU
	// readback system backing GetCachedGroundHeight() -- the two things left once the GPU image/
	// view/sampler ownership, the CPU noise-based paging/generation machinery, and the noise-based
	// CPU height sampler (SampleTerrain -- it queried an unrelated noise function, not what's
	// actually rendered) were all deleted. The terrain arrays are now normal pass-owned graph
	// resources, Created<> by TerrainGenNode; see terrain/TerrainClipmap.hpp's git history for
	// what used to live here. Named for what it actually does now, not what it used to be.
	class TerrainManager: public ITerrainManager {
	public:
		TerrainManager() = default;
		~TerrainManager() override = default;

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
			vk::Device   device,
			VmaAllocator allocator,
			vk::Queue    transferQueue,
			uint32_t     transferQueueFamily,
			uint32_t     numLODs = constants::Class::Terrain::DefaultMaxLODs,
			float        baseTexelSize = constants::Class::Terrain::BaseTexelSize
		);
		void Cleanup();

		void Regenerate() override { m_forceRegenerate = true; }
		bool ShouldForceRegeneration() { return m_forceRegenerate; }
		void ResetForceRegeneration() { m_forceRegenerate = false; }

		uint32_t GetNumLODs() const { return numLODs; }

		float GetBaseTexelSize() const { return baseTexelSize; }

		// Async GPU->CPU readback -- a real device-local image region (TerrainMinMaxTexture's
		// coarse mip, in practice) copied back through a dedicated transfer-queue command buffer
		// + timeline semaphore + host-visible staging buffer, polled non-blockingly rather than
		// waited on. Moved here verbatim from Engine (which owned it directly before this class
		// had anything worth attaching it to); the caller still decides *what* region to request
		// (that needs camera position + the physical registry, neither of which this class owns).
		bool TriggerImageRegionReadbackAsync(
			vk::Image       image,
			uint32_t        arrayLayer,
			uint32_t        mipLevel,
			vk::Offset2D    offset,
			vk::Extent2D    extent,
			vk::ImageLayout currentLayout = vk::ImageLayout::eGeneral
		);

		void PollReadbackData();

		// The real cached-ground-height query: max of .g (the max-height channel) over whatever
		// region the last completed readback covered, or fallback if nothing has completed yet.
		// RG32F texels (min, max), not a raw height sample -- see TriggerImageRegionReadbackAsync's
		// real caller (Engine::UpdateCamera) for why this reads a min/max mip instead of the
		// heightmap directly.
		float GetCachedGroundHeight(float fallback) const;

	private:
		uint32_t numLODs{constants::Class::Terrain::DefaultMaxLODs};
		float    baseTexelSize{0.5f};
		bool     m_forceRegenerate{true};

		vk::Device   device{nullptr};
		VmaAllocator allocator{VK_NULL_HANDLE};
		vk::Queue    transferQueue{nullptr};
		uint32_t     transferQueueFamily{0};

		vk::CommandPool   asyncTransferCommandPool{nullptr};
		vk::CommandBuffer asyncTransferCommandBuffer{nullptr};
		vk::Buffer        readbackStagingBuffer{nullptr};
		VmaAllocation     readbackStagingAllocation{VK_NULL_HANDLE};
		void*             readbackStagingMapped{nullptr};
		vk::Semaphore     readbackTimelineSemaphore{nullptr};
		uint64_t          readbackSubmittedTimelineValue{0};
		uint64_t          readbackCompletedTimelineValue{0};
		bool              readbackInFlight{false};

		std::vector<glm::vec2> cachedReadbackData;
		uint32_t                cachedReadbackWidth{0};
		uint32_t                cachedReadbackHeight{0};
		bool                    hasReadbackData{false};
	};

	// mips = 1: this image has no mip chain of its own -- only TerrainMinMaxDesc below does.
	inline graph::ResourceDesc TerrainClipmapDesc(std::uint32_t numLODs) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = constants::Class::Terrain::MapDim,
			.height = constants::Class::Terrain::MapDim,
			.mips = 1,
			.layers = numLODs,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32B32A32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc
			),
			.persistent = true,
		};
	}

	// floor(log2(MapDim)) + 1 -- the full mip chain down to 1x1 for a 1024-wide image (mips
	// 0-10), well under SPD's 12-mip-per-dispatch limit. RG32F, not RGBA32F: only min (x) and
	// max (y) are meaningful, and the per-mip storage views this needs (PhysicalTexture::
	// CreateView, PhysicalResource.hpp) are what the broken mips=1 claim on this key used to mask
	// -- nothing ever read past mip 0 before.
	inline graph::ResourceDesc TerrainMinMaxDesc(std::uint32_t numLODs) {
		return graph::ResourceDesc{
			.kind = graph::ResourceDesc::Kind::Image2D,
			.width = constants::Class::Terrain::MapDim,
			.height = constants::Class::Terrain::MapDim,
			.mips = 11,
			.layers = numLODs,
			.formatCode = static_cast<std::uint32_t>(vk::Format::eR32G32Sfloat),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc
			),
			.persistent = true,
		};
	}

	inline graph::ResourceDesc TerrainBiomeDesc(std::uint32_t numLODs) {
		return TerrainClipmapDesc(numLODs);
	}

} // namespace brassica
