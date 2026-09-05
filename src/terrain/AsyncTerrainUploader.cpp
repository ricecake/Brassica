#include "terrain/AsyncTerrainUploader.hpp"
#include <cstring>
#include "spdlog/spdlog.h"

namespace brassica {

	AsyncTerrainUploader::~AsyncTerrainUploader() {
		Cleanup();
	}

	void AsyncTerrainUploader::Init(
		vk::Device dev,
		VmaAllocator alloc,
		uint32_t queueFamilyIdx,
		uint32_t maxConcurrentUploads
	) {
		device = dev;
		allocator = alloc;

		vk::CommandPoolCreateInfo poolInfo{};
		poolInfo.setQueueFamilyIndex(queueFamilyIdx);
		poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
		commandPool = device.createCommandPool(poolInfo);

		requests.resize(maxConcurrentUploads);
		vk::FenceCreateInfo fenceInfo{};

		for (auto& req : requests) {
			req.fence = device.createFence(fenceInfo);

			vk::CommandBufferAllocateInfo cmdInfo{};
			cmdInfo.setCommandPool(commandPool);
			cmdInfo.setLevel(vk::CommandBufferLevel::ePrimary);
			cmdInfo.setCommandBufferCount(1);
			req.commandBuffer = device.allocateCommandBuffers(cmdInfo).front();
			req.inFlight = false;
		}
	}

	void AsyncTerrainUploader::Cleanup() {
		if (!device) return;

		// Wait for any remaining fences before cleanup
		for (auto& req : requests) {
			if (req.inFlight && req.fence) {
				(void)device.waitForFences(req.fence, VK_TRUE, 100000000);
			}
			if (req.stagingBuffer && req.stagingAllocation) {
				vmaDestroyBuffer(allocator, req.stagingBuffer, req.stagingAllocation);
				req.stagingBuffer = nullptr;
				req.stagingAllocation = VK_NULL_HANDLE;
			}
			if (req.fence) {
				device.destroyFence(req.fence);
				req.fence = nullptr;
			}
		}

		if (commandPool) {
			device.destroyCommandPool(commandPool);
			commandPool = nullptr;
		}
	}

	void AsyncTerrainUploader::Poll() {
		for (auto& req : requests) {
			if (!req.inFlight) continue;

			vk::Result status = device.getFenceStatus(req.fence);
			if (status == vk::Result::eSuccess) {
				// Upload completed without blocking
				device.resetFences(req.fence);
				if (req.stagingBuffer && req.stagingAllocation) {
					vmaDestroyBuffer(allocator, req.stagingBuffer, req.stagingAllocation);
					req.stagingBuffer = nullptr;
					req.stagingAllocation = VK_NULL_HANDLE;
				}
				req.inFlight = false;
			}
		}
	}

	bool AsyncTerrainUploader::HasInFlightUploads() const {
		for (const auto& req : requests) {
			if (req.inFlight) return true;
		}
		return false;
	}

	bool AsyncTerrainUploader::UploadLevelAsync(
		uint32_t levelIndex,
		std::span<const glm::vec4> data,
		vk::Image targetImage,
		uint32_t width,
		uint32_t height,
		vk::Queue transferQueue
	) {
		Poll(); // Reclaim completed uploads first

		// Find available request slot
		PendingUploadRequest* slot = nullptr;
		for (auto& req : requests) {
			if (!req.inFlight) {
				slot = &req;
				break;
			}
		}

		if (!slot) {
			spdlog::warn("AsyncTerrainUploader: No available upload slot for LOD level {}", levelIndex);
			return false; // Queue full
		}

		VkDeviceSize bufferSize = data.size_bytes();

		// Allocate staging buffer
		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = bufferSize;
		bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer vkBuf = VK_NULL_HANDLE;
		VmaAllocationInfo resultAllocInfo{};
		if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &vkBuf, &slot->stagingAllocation, &resultAllocInfo) != VK_SUCCESS) {
			spdlog::error("Failed to create staging buffer for terrain upload.");
			return false;
		}
		slot->stagingBuffer = vkBuf;

		// Copy data to mapped staging memory
		std::memcpy(resultAllocInfo.pMappedData, data.data(), bufferSize);

		// Record copy commands
		slot->commandBuffer.reset();
		vk::CommandBufferBeginInfo cmdBegin{vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		slot->commandBuffer.begin(cmdBegin);

		// Transition target image layer to DST_OPTIMAL
		vk::ImageMemoryBarrier2 barrier1{};
		barrier1.setSrcStageMask(vk::PipelineStageFlagBits2::eAllCommands);
		barrier1.setSrcAccessMask(vk::AccessFlagBits2::eNone);
		barrier1.setDstStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrier1.setDstAccessMask(vk::AccessFlagBits2::eTransferWrite);
		barrier1.setOldLayout(vk::ImageLayout::eUndefined);
		barrier1.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
		barrier1.setImage(targetImage);
		barrier1.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, levelIndex, 1));

		vk::DependencyInfo depInfo1{};
		depInfo1.setImageMemoryBarriers(barrier1);
		slot->commandBuffer.pipelineBarrier2(depInfo1);

		// Copy buffer to image layer
		vk::BufferImageCopy copyRegion{};
		copyRegion.setBufferOffset(0);
		copyRegion.setBufferRowLength(width);
		copyRegion.setBufferImageHeight(height);
		copyRegion.setImageSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, levelIndex, 1));
		copyRegion.setImageOffset(vk::Offset3D{0, 0, 0});
		copyRegion.setImageExtent(vk::Extent3D{width, height, 1});

		slot->commandBuffer.copyBufferToImage(slot->stagingBuffer, targetImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

		// Transition target image layer to SHADER_READ_ONLY_OPTIMAL
		vk::ImageMemoryBarrier2 barrier2{};
		barrier2.setSrcStageMask(vk::PipelineStageFlagBits2::eTransfer);
		barrier2.setSrcAccessMask(vk::AccessFlagBits2::eTransferWrite);
		barrier2.setDstStageMask(vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eTaskShaderEXT | vk::PipelineStageFlagBits2::eFragmentShader);
		barrier2.setDstAccessMask(vk::AccessFlagBits2::eShaderRead);
		barrier2.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
		barrier2.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
		barrier2.setImage(targetImage);
		barrier2.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, levelIndex, 1));

		vk::DependencyInfo depInfo2{};
		depInfo2.setImageMemoryBarriers(barrier2);
		slot->commandBuffer.pipelineBarrier2(depInfo2);

		slot->commandBuffer.end();

		// Submit command buffer non-blockingly with fence
		vk::CommandBufferSubmitInfo cmdSubmit{};
		cmdSubmit.setCommandBuffer(slot->commandBuffer);

		vk::SubmitInfo2 submitInfo{};
		submitInfo.setCommandBufferInfos(cmdSubmit);

		slot->levelIndex = levelIndex;
		slot->inFlight = true;

		transferQueue.submit2(submitInfo, slot->fence);

		return true;
	}

} // namespace brassica
