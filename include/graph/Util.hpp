#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <span>

#include <vulkan/vulkan.hpp>

#include "graph/PhysicalResource.hpp"
#include "vk_mem_alloc.h"

namespace brassica::graph {
	class PhysicalResourceRegistry;
}

namespace brassica::utils {

	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc = {},
		vk::Queue                        transferQueue = {}
	);

	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		vk::Device                       device,
		VmaAllocator                     allocator,
		vk::Queue                        transferQueue,
		vk::CommandPool                  transientPool,
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc
	);

	template <graph::ResourceRef Key>
	std::shared_ptr<graph::PhysicalTexture> CreateAndRegisterStaticTexture(
		graph::PhysicalResourceRegistry& registry,
		std::span<const std::uint8_t>    pixelData,
		const graph::ResourceDesc&       desc,
		vk::ImageLayout                  targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::Queue                        transferQueue = {}
	);

} // namespace brassica::utils

#include "graph/PhysicalRegistry.hpp"

namespace brassica::utils {

	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc,
		vk::Queue                        transferQueue
	) {
		(void)transferQueue;
		registry.UploadPredefinedBuffer(graph::IdOf<Key>(), data.data(), data.size_bytes(), desc);
		return registry.GetBuffer<Key>();
	}

	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		vk::Device                       device,
		VmaAllocator                     allocator,
		vk::Queue                        transferQueue,
		vk::CommandPool                  transientPool,
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc
	) {
		(void)device;
		(void)allocator;
		(void)transientPool;
		return CreateAndRegisterStaticBuffer<Key, T>(registry, data, desc, transferQueue);
	}

	template <graph::ResourceRef Key>
	std::shared_ptr<graph::PhysicalTexture> CreateAndRegisterStaticTexture(
		graph::PhysicalResourceRegistry& registry,
		std::span<const std::uint8_t>    pixelData,
		const graph::ResourceDesc&       desc,
		vk::ImageLayout                  targetLayout,
		vk::Queue                        transferQueue
	) {
		(void)transferQueue;
		registry.UploadPredefinedTexture(
			graph::IdOf<Key>(),
			pixelData.data(),
			pixelData.size_bytes(),
			desc,
			static_cast<std::uint32_t>(targetLayout)
		);
		return registry.GetTexture<Key>();
	}

} // namespace brassica::utils

namespace brassica::graph {

	inline void PhysicalResourceRegistry::UploadPredefinedBuffer(
		ResourceId          id,
		const void*         data,
		std::size_t         sizeBytes,
		const ResourceDesc& desc
	) {
		vk::Device   device = GetDevice();
		VmaAllocator allocator = GetAllocator();
		vk::Queue    queue = GetQueue();

		const ResourceId resolvedId = ResolveId(id);
		auto             destBuffer = GetBuffer(resolvedId);
		if (!destBuffer) {
			ResourceDesc finalDesc = desc;
			if (finalDesc.byteSize == 0) {
				finalDesc = StorageBufferDesc(sizeBytes);
			}
			destBuffer = std::make_shared<PhysicalBuffer>(device, allocator, finalDesc);
			m_buffers[resolvedId] = destBuffer;
		}
		destBuffer->SetHasDefinedContents(true);

		if (sizeBytes > 0 && allocator != VK_NULL_HANDLE) {
			VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = sizeBytes;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

			VmaAllocationCreateInfo allocInfo = {};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          rawStagingBuffer = VK_NULL_HANDLE;
			VmaAllocation     stagingAllocation = nullptr;
			VmaAllocationInfo vmaAllocInfo{};
			vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &rawStagingBuffer, &stagingAllocation, &vmaAllocInfo);

			if (vmaAllocInfo.pMappedData) {
				std::memcpy(vmaAllocInfo.pMappedData, data, sizeBytes);
			}

			if (device && queue) {
				vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlagBits::eTransient, 0};
				vk::CommandPool           transientPool = device.createCommandPool(poolInfo);

				vk::CommandBufferAllocateInfo cmdAllocInfo{transientPool, vk::CommandBufferLevel::ePrimary, 1};
				vk::CommandBuffer             cmd = device.allocateCommandBuffers(cmdAllocInfo).front();

				cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
				vk::BufferCopy copyRegion{0, 0, sizeBytes};
				cmd.copyBuffer(rawStagingBuffer, destBuffer->GetBuffer(), 1, &copyRegion);
				cmd.end();

				vk::SubmitInfo submitInfo{};
				submitInfo.setCommandBuffers(cmd);
				(void)queue.submit(1, &submitInfo, nullptr);
				queue.waitIdle();

				device.destroyCommandPool(transientPool);
			}

			vmaDestroyBuffer(allocator, rawStagingBuffer, stagingAllocation);
		}

		m_buffers[ResolveId(id)] = destBuffer;
	}

	inline void PhysicalResourceRegistry::UploadPredefinedTexture(
		ResourceId          id,
		const void*         pixelData,
		std::size_t         sizeBytes,
		const ResourceDesc& desc,
		std::uint32_t       targetLayout
	) {
		vk::Device   device = GetDevice();
		VmaAllocator allocator = GetAllocator();
		vk::Queue    queue = GetQueue();
		auto         layout = static_cast<vk::ImageLayout>(targetLayout);

		const ResourceId resolvedId = ResolveId(id);
		auto             destTexture = GetTexture(resolvedId);
		ResourceDesc texDesc;
		if (!destTexture) {
			texDesc = desc;
			if (texDesc.usageMask == 0) {
				texDesc.usageMask = static_cast<std::uint32_t>(
					vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
				);
			}
			destTexture = std::make_shared<PhysicalTexture>(device, allocator, texDesc);
			AssignAndWriteBindlessIndices(*destTexture);
			m_textures[resolvedId] = destTexture;
		}
		destTexture->SetHasDefinedContents(true);

		if (sizeBytes > 0 && allocator != VK_NULL_HANDLE && device && queue) {
			VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
			bufferInfo.size = sizeBytes;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

			VmaAllocationCreateInfo allocInfo = {};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBuffer          rawStagingBuffer = VK_NULL_HANDLE;
			VmaAllocation     stagingAllocation = nullptr;
			VmaAllocationInfo vmaAllocInfo{};
			vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &rawStagingBuffer, &stagingAllocation, &vmaAllocInfo);

			if (vmaAllocInfo.pMappedData) {
				std::memcpy(vmaAllocInfo.pMappedData, pixelData, sizeBytes);
			}

			vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlagBits::eTransient, 0};
			vk::CommandPool           transientPool = device.createCommandPool(poolInfo);

			vk::CommandBufferAllocateInfo cmdAllocInfo{transientPool, vk::CommandBufferLevel::ePrimary, 1};
			vk::CommandBuffer             cmd = device.allocateCommandBuffers(cmdAllocInfo).front();

			cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

			vk::ImageSubresourceRange subresourceRange{
				vk::ImageAspectFlagBits::eColor,
				0,
				texDesc.mips ? texDesc.mips : 1,
				0,
				texDesc.layers ? texDesc.layers : 1
			};

			vk::ImageMemoryBarrier barrier1{};
			barrier1.setOldLayout(vk::ImageLayout::eUndefined)
				.setNewLayout(vk::ImageLayout::eTransferDstOptimal)
				.setSrcAccessMask({})
				.setDstAccessMask(vk::AccessFlagBits::eTransferWrite)
				.setImage(destTexture->GetImage())
				.setSubresourceRange(subresourceRange);

			cmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::PipelineStageFlagBits::eTransfer,
				{},
				{},
				{},
				barrier1
			);

			vk::BufferImageCopy copyRegion{};
			copyRegion.setBufferOffset(0)
				.setBufferRowLength(0)
				.setBufferImageHeight(0)
				.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
				.setImageOffset({0, 0, 0})
				.setImageExtent({texDesc.width, texDesc.height, std::max(1u, texDesc.depth)});

			cmd.copyBufferToImage(
				rawStagingBuffer,
				destTexture->GetImage(),
				vk::ImageLayout::eTransferDstOptimal,
				copyRegion
			);

			vk::ImageMemoryBarrier barrier2{};
			barrier2.setOldLayout(vk::ImageLayout::eTransferDstOptimal)
				.setNewLayout(layout)
				.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
				.setDstAccessMask(vk::AccessFlagBits::eShaderRead)
				.setImage(destTexture->GetImage())
				.setSubresourceRange(subresourceRange);

			cmd.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eAllCommands,
				{},
				{},
				{},
				barrier2
			);

			cmd.end();

			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(cmd);
			(void)queue.submit(1, &submitInfo, nullptr);
			queue.waitIdle();

			device.destroyCommandPool(transientPool);
			vmaDestroyBuffer(allocator, rawStagingBuffer, stagingAllocation);
		}

		AssignAndWriteBindlessIndices(*destTexture);
		m_textures[ResolveId(id)] = destTexture;
	}

	inline void PhysicalResourceRegistry::WaitIdle() {
		if (m_device) {
			m_device.waitIdle();
		}
	}

} // namespace brassica::graph
