#pragma once

#include <memory>
#include <span>
#include <vector>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "vk_mem_alloc.h"

namespace brassica {

	struct PendingUploadRequest {
		vk::CommandBuffer commandBuffer{nullptr};
		vk::Buffer        stagingBuffer{nullptr};
		VmaAllocation     stagingAllocation{VK_NULL_HANDLE};
		uint32_t          levelIndex{0};
		bool              inFlight{false};
		uint64_t          targetTimelineValue = 0;
	};

	class AsyncTerrainUploader {
	public:
		AsyncTerrainUploader() = default;
		~AsyncTerrainUploader();

		void Init(vk::Device dev, VmaAllocator alloc, uint32_t queueFamilyIdx, uint32_t maxConcurrentUploads = 32);
		void Cleanup();

		// Non-blocking upload request for a clipmap layer
		bool UploadLevelAsync(
			uint32_t                   levelIndex,
			std::span<const glm::vec4> data,
			vk::Image                  targetImage,
			uint32_t                   width,
			uint32_t                   height,
			vk::Queue                  transferQueue
		);

		// Non-blocking upload request for sub-regions of a clipmap layer using vk::BufferImageCopy
		bool UploadRegionAsync(
			uint32_t                             levelIndex,
			std::span<const glm::vec4>           data,
			std::span<const vk::BufferImageCopy> regions,
			vk::Image                            targetImage,
			vk::Queue                            transferQueue
		);

		std::vector<vk::SemaphoreSubmitInfo> GetWaitSemaphores() const;

		// Non-blocking poll to reclaim finished staging buffers and fences
		void Poll();

		// Check if any uploads are currently in-flight
		bool HasInFlightUploads() const;

	private:
		vk::Device      device{nullptr};
		VmaAllocator    allocator{VK_NULL_HANDLE};
		vk::CommandPool commandPool{nullptr};
		vk::Semaphore   timelineSemaphore;
		uint64_t        currentTimelineCounter = 0;

		std::vector<PendingUploadRequest> requests;
	};

} // namespace brassica
