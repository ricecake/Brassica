#include "terrain/TerrainManager.hpp"

#include <algorithm>
#include <cstring>

#include "spdlog/spdlog.h"

namespace brassica {

	void TerrainManager::Init(
		vk::Device   dev,
		VmaAllocator alloc,
		vk::Queue    transferQ,
		uint32_t     transferQFamily,
		uint32_t     lods,
		float        baseTexel
	) {
		device = dev;
		allocator = alloc;
		transferQueue = transferQ;
		transferQueueFamily = transferQFamily;
		numLODs = lods;
		baseTexelSize = baseTexel;

		if (!device) {
			return;
		}

		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.setQueueFamilyIndex(transferQueueFamily);
		poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
		asyncTransferCommandPool = device.createCommandPool(poolInfo);

		vk::CommandBufferAllocateInfo allocInfo{};
		allocInfo.setCommandPool(asyncTransferCommandPool);
		allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
		allocInfo.setCommandBufferCount(1);
		asyncTransferCommandBuffer = device.allocateCommandBuffers(allocInfo).front();

		vk::SemaphoreTypeCreateInfo typeInfo{};
		typeInfo.setSemaphoreType(vk::SemaphoreType::eTimeline);
		typeInfo.setInitialValue(0);

		vk::SemaphoreCreateInfo semInfo{};
		semInfo.setPNext(&typeInfo);
		readbackTimelineSemaphore = device.createSemaphore(semInfo);

		// Comfortably covers the small (e.g. 2x2 RG32F = 16 bytes) coarse-mip readback this is
		// actually used for -- generous rather than tight-fitted, matching the staging buffers
		// elsewhere in this codebase.
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = 1024 * 1024; // 1MB max readback staging
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocCreateInfo.flags =
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          buf{VK_NULL_HANDLE};
		VmaAllocationInfo allocationInfo{};
		if (vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo, &buf, &readbackStagingAllocation, &allocationInfo) ==
		    VK_SUCCESS) {
			readbackStagingBuffer = buf;
			readbackStagingMapped = allocationInfo.pMappedData;
		} else {
			spdlog::error("TerrainManager: failed to allocate readback staging buffer.");
		}
	}

	void TerrainManager::Cleanup() {
		if (!device) {
			return;
		}
		if (readbackStagingBuffer && readbackStagingAllocation) {
			vmaDestroyBuffer(allocator, readbackStagingBuffer, readbackStagingAllocation);
			readbackStagingBuffer = nullptr;
			readbackStagingAllocation = VK_NULL_HANDLE;
			readbackStagingMapped = nullptr;
		}
		if (readbackTimelineSemaphore) {
			device.destroySemaphore(readbackTimelineSemaphore);
			readbackTimelineSemaphore = nullptr;
		}
		if (asyncTransferCommandPool) {
			device.destroyCommandPool(asyncTransferCommandPool);
			asyncTransferCommandPool = nullptr;
		}
	}

	bool TerrainManager::TriggerImageRegionReadbackAsync(
		vk::Image       image,
		uint32_t        arrayLayer,
		uint32_t        mipLevel,
		vk::Offset2D    offset,
		vk::Extent2D    extent,
		vk::ImageLayout currentLayout
	) {
		if (!image || readbackInFlight || !readbackStagingBuffer) {
			return false;
		}

		size_t requiredBytes = static_cast<size_t>(extent.width) * extent.height * sizeof(glm::vec2);
		if (requiredBytes > 1024 * 1024) {
			spdlog::warn("TerrainManager: requested readback size {} bytes exceeds staging capacity.", requiredBytes);
			return false;
		}

		cachedReadbackWidth = extent.width;
		cachedReadbackHeight = extent.height;

		readbackSubmittedTimelineValue++;

		asyncTransferCommandBuffer.reset();
		asyncTransferCommandBuffer.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		vk::ImageMemoryBarrier2 barrier1{};
		barrier1.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands);
		barrier1.setSrcAccessMask(vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead);
		barrier1.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrier1.setDstAccessMask(vk::AccessFlagBits2::eTransferRead);
		barrier1.setOldLayout(currentLayout);
		barrier1.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrier1.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
		barrier1.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
		barrier1.setImage(image);
		barrier1.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mipLevel, 1, arrayLayer, 1});

		vk::DependencyInfo depInfo1{};
		depInfo1.setImageMemoryBarriers(barrier1);
		asyncTransferCommandBuffer.pipelineBarrier2(depInfo1);

		vk::BufferImageCopy copyRegion{};
		copyRegion.setBufferOffset(0);
		copyRegion.setBufferRowLength(extent.width);
		copyRegion.setBufferImageHeight(extent.height);
		copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mipLevel, arrayLayer, 1});
		copyRegion.setImageOffset(vk::Offset3D{offset.x, offset.y, 0});
		copyRegion.setImageExtent(vk::Extent3D{extent.width, extent.height, 1});

		asyncTransferCommandBuffer
			.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, readbackStagingBuffer, copyRegion);

		vk::ImageMemoryBarrier2 barrier2{};
		barrier2.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrier2.setSrcAccessMask(vk::AccessFlagBits2::eTransferRead);
		barrier2.setDstStageMask(vk::PipelineStageFlagBits2::eAllCommands);
		barrier2.setDstAccessMask(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
		barrier2.setOldLayout(vk::ImageLayout::eTransferSrcOptimal);
		barrier2.setNewLayout(currentLayout);
		barrier2.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
		barrier2.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
		barrier2.setImage(image);
		barrier2.setSubresourceRange(vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, mipLevel, 1, arrayLayer, 1});

		vk::DependencyInfo depInfo2{};
		depInfo2.setImageMemoryBarriers(barrier2);
		asyncTransferCommandBuffer.pipelineBarrier2(depInfo2);

		asyncTransferCommandBuffer.end();

		vk::SemaphoreSubmitInfo signalInfo{};
		signalInfo.setSemaphore(readbackTimelineSemaphore);
		signalInfo.setValue(readbackSubmittedTimelineValue);
		signalInfo.setStageMask(vk::PipelineStageFlagBits2::eAllTransfer);

		vk::CommandBufferSubmitInfo cmdSubmitInfo{};
		cmdSubmitInfo.setCommandBuffer(asyncTransferCommandBuffer);

		vk::SubmitInfo2 submitInfo{};
		submitInfo.setCommandBufferInfos(cmdSubmitInfo);
		submitInfo.setSignalSemaphoreInfos(signalInfo);

		transferQueue.submit2(submitInfo, nullptr);
		readbackInFlight = true;

		return true;
	}

	void TerrainManager::PollReadbackData() {
		if (!readbackInFlight || !readbackTimelineSemaphore) {
			return;
		}

		uint64_t currentValue = device.getSemaphoreCounterValue(readbackTimelineSemaphore);
		if (currentValue >= readbackSubmittedTimelineValue) {
			readbackCompletedTimelineValue = currentValue;
			readbackInFlight = false;

			size_t pixelCount = static_cast<size_t>(cachedReadbackWidth) * cachedReadbackHeight;
			cachedReadbackData.resize(pixelCount);
			if (readbackStagingMapped) {
				std::memcpy(cachedReadbackData.data(), readbackStagingMapped, pixelCount * sizeof(glm::vec2));
			}
			hasReadbackData = true;
		}
	}

	float TerrainManager::GetCachedGroundHeight(float fallback) const {
		if (!hasReadbackData || cachedReadbackData.empty()) {
			return fallback;
		}
		float maxHeight = fallback;
		for (const glm::vec2& sample : cachedReadbackData) {
			maxHeight = std::max(maxHeight, sample.y); // .y = max height, from the min/max mip texel
		}
		return maxHeight;
	}

} // namespace brassica
