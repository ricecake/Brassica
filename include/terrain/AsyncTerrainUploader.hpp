#pragma once

#include <span>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	struct PendingUploadRequest {
		vk::CommandBuffer commandBuffer{nullptr};
		vk::Fence         fence{nullptr};
		vk::Buffer        stagingBuffer{nullptr};
		VmaAllocation     stagingAllocation{VK_NULL_HANDLE};
		uint32_t          levelIndex{0};
		bool              inFlight{false};
	};

	class AsyncTerrainUploader {
	public:
		AsyncTerrainUploader() = default;
		~AsyncTerrainUploader();

		void Init(
			vk::Device dev,
			VmaAllocator alloc,
			uint32_t queueFamilyIdx,
			uint32_t maxConcurrentUploads = 8
		);
		void Cleanup();

		// Non-blocking upload request for a clipmap layer
		bool UploadLevelAsync(
			uint32_t levelIndex,
			std::span<const glm::vec4> data,
			vk::Image targetImage,
			uint32_t width,
			uint32_t height,
			vk::Queue transferQueue
		);

		// Non-blocking poll to reclaim finished staging buffers and fences
		void Poll();

		// Check if any uploads are currently in-flight
		bool HasInFlightUploads() const;

	private:
		vk::Device      device{nullptr};
		VmaAllocator    allocator{VK_NULL_HANDLE};
		vk::CommandPool commandPool{nullptr};

		std::vector<PendingUploadRequest> requests;
	};

} // namespace brassica
