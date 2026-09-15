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

	// Both helpers upload through registry.GetQueue() -- there is no way to route a call
	// through a different queue than the one the registry was constructed/wired with, so
	// callers should not read a "transfer queue" parameter into these; there isn't one.
	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc = {}
	);

	template <graph::ResourceRef Key>
	std::shared_ptr<graph::PhysicalTexture> CreateAndRegisterStaticTexture(
		graph::PhysicalResourceRegistry& registry,
		std::span<const std::uint8_t>    pixelData,
		const graph::ResourceDesc&       desc,
		vk::ImageLayout                  targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal
	);

} // namespace brassica::utils

#include "graph/PhysicalRegistry.hpp"

namespace brassica::utils {

	template <graph::ResourceRef Key, typename T>
	std::shared_ptr<graph::PhysicalBuffer> CreateAndRegisterStaticBuffer(
		graph::PhysicalResourceRegistry& registry,
		std::span<const T>               data,
		const graph::ResourceDesc&       desc
	) {
		registry.UploadBufferImmediate(graph::IdOf<Key>(), data.data(), data.size_bytes(), desc);
		return registry.GetBuffer<Key>();
	}

	template <graph::ResourceRef Key>
	std::shared_ptr<graph::PhysicalTexture> CreateAndRegisterStaticTexture(
		graph::PhysicalResourceRegistry& registry,
		std::span<const std::uint8_t>    pixelData,
		const graph::ResourceDesc&       desc,
		vk::ImageLayout                  targetLayout
	) {
		registry
			.UploadTextureImmediate(graph::IdOf<Key>(), pixelData.data(), pixelData.size_bytes(), desc, targetLayout);
		return registry.GetTexture<Key>();
	}

} // namespace brassica::utils

namespace brassica::graph {

	// Pre-frame-loop only: a real fence, not queue.waitIdle() -- unlike the old
	// UploadPredefinedBuffer this replaces, waitIdle would stall the *whole* queue (every other
	// in-flight frame, if this ever ran after the frame loop had started, which it doesn't
	// today) for what only needs to wait on this one transfer.
	inline void PhysicalResourceRegistry::UploadBufferImmediate(
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
				std::memcpy(vmaAllocInfo.pMappedData, data, sizeBytes);
			}

			vk::CommandPoolCreateInfo poolInfo{vk::CommandPoolCreateFlagBits::eTransient, 0};
			vk::CommandPool           transientPool = device.createCommandPool(poolInfo);

			vk::CommandBufferAllocateInfo cmdAllocInfo{transientPool, vk::CommandBufferLevel::ePrimary, 1};
			vk::CommandBuffer             cmd = device.allocateCommandBuffers(cmdAllocInfo).front();

			cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			vk::BufferCopy copyRegion{0, 0, sizeBytes};
			cmd.copyBuffer(rawStagingBuffer, destBuffer->GetBuffer(), 1, &copyRegion);
			cmd.end();

			vk::Fence      fence = device.createFence(vk::FenceCreateInfo{});
			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(cmd);
			(void)queue.submit(1, &submitInfo, fence);
			(void)device.waitForFences(fence, VK_TRUE, UINT64_MAX);
			device.destroyFence(fence);

			device.destroyCommandPool(transientPool);
			vmaDestroyBuffer(allocator, rawStagingBuffer, stagingAllocation);
		}

		m_buffers[resolvedId] = destBuffer;
	}

	inline void PhysicalResourceRegistry::UploadTextureImmediate(
		ResourceId          id,
		const void*         pixelData,
		std::size_t         sizeBytes,
		const ResourceDesc& desc,
		vk::ImageLayout     targetLayout
	) {
		vk::Device   device = GetDevice();
		VmaAllocator allocator = GetAllocator();
		vk::Queue    queue = GetQueue();

		const ResourceId resolvedId = ResolveId(id);
		auto             destTexture = GetTexture(resolvedId);
		ResourceDesc     texDesc = desc;
		if (!destTexture) {
			if (texDesc.usageMask == 0) {
				texDesc.usageMask = static_cast<std::uint32_t>(
					vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
				);
			}
			destTexture = std::make_shared<PhysicalTexture>(device, allocator, texDesc);
			AssignAndWriteBindlessIndices(*destTexture);
			m_textures[resolvedId] = destTexture;
		} else {
			// Re-upload into an existing texture: texDesc must describe what that texture
			// actually *is*, not the caller's desc -- the caller may pass a stale or
			// default-constructed one on this path, and the subresource range/copy extent below
			// need the real width/height/mips/layers or they silently copy the wrong region.
			texDesc = destTexture->GetDesc();
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
				.setNewLayout(targetLayout)
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

			// The two manual barriers above are real, but until now nothing told the registry's
			// own tracking they happened -- destTexture->GetCurrentLayout() kept reporting
			// eUndefined (or whatever it was before this call) forever after, silently
			// desynced from the image's real layout. Only the final state matters here: barrier1's
			// transitional eTransferDstOptimal is never observed by anything outside this
			// function, which waits on the fence below before returning.
			destTexture->SetCurrentLayout(targetLayout);
			destTexture->SetLastStageAccess(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderRead);

			cmd.end();

			vk::Fence      fence = device.createFence(vk::FenceCreateInfo{});
			vk::SubmitInfo submitInfo{};
			submitInfo.setCommandBuffers(cmd);
			(void)queue.submit(1, &submitInfo, fence);
			(void)device.waitForFences(fence, VK_TRUE, UINT64_MAX);
			device.destroyFence(fence);

			device.destroyCommandPool(transientPool);
			vmaDestroyBuffer(allocator, rawStagingBuffer, stagingAllocation);
		}

		AssignAndWriteBindlessIndices(*destTexture);
		m_textures[resolvedId] = destTexture;
	}

	inline void PhysicalResourceRegistry::WaitIdle() {
		if (m_device) {
			m_device.waitIdle();
		}
	}

	inline std::span<std::byte> PhysicalResourceRegistry::BeginHostWrite(
		ResourceId    id,
		std::size_t   sizeBytes,
		std::uint64_t frameIndex,
		bool          insideRendering
	) {
		const ResourceId resolvedId = ResolveId(id);

		// find()-based, not operator[]-based: a fresh entry must not compare equal to a real
		// frameIndex of 0 -- operator[]'s default-constructed 0 did exactly that, misfiring on
		// every resource's very first write (frame 0 looked like a repeat of "frame 0").
		const auto existingFrameIt = m_hostWriteFrame.find(resolvedId);
		if (existingFrameIt != m_hostWriteFrame.end() && existingFrameIt->second == frameIndex) {
			throw std::runtime_error(
				"BeginHostWrite: a second write to '" + std::string(resolvedId->name) +
				"' was attempted in the same frame -- one CPU write per resource per frame is the "
				"contract; a second write would silently clobber the first."
			);
		}

		auto beginStaged = [&](const ResourceDesc& desc) -> std::span<std::byte> {
			if (insideRendering) {
				throw std::runtime_error(
					"BeginHostWrite: a Staged write to '" + std::string(resolvedId->name) +
					"' was attempted from inside a render pass -- Staged writes record a real copy "
					"command, which vkCmdCopyBuffer/vkCmdCopyBufferToImage can't do between "
					"vkCmdBeginRendering/vkCmdEndRendering. Use HostAccess::Mapped for this "
					"resource instead, or move the write to a node whose domain isn't Graphics."
				);
			}
			(void)desc;
			StagingSlot& slot = AcquireStagingSlot(resolvedId, sizeBytes, frameIndex);
			m_hostWriteFrame[resolvedId] = frameIndex;
			m_pendingHostWrites[resolvedId] = {frameIndex, sizeBytes};
			return {static_cast<std::byte*>(slot.mapped), sizeBytes};
		};

		if (auto buf = GetBuffer(resolvedId)) {
			const ResourceDesc& desc = buf->GetDesc();
			if (desc.hostAccess == HostAccess::Mapped) {
				if (sizeBytes > buf->SliceStride()) {
					throw std::runtime_error(
						"BeginHostWrite: write of " + std::to_string(sizeBytes) + " bytes to '" +
						std::string(resolvedId->name) + "' exceeds its provisioned per-slot capacity of " +
						std::to_string(buf->SliceStride()) +
						" bytes -- the realization that provisioned it must request at least this much."
					);
				}
				void* slice = buf->MappedSlice(frameIndex);
				if (!slice) {
					return {};
				}
				m_hostWriteFrame[resolvedId] = frameIndex;
				m_pendingHostWrites[resolvedId] = {frameIndex, sizeBytes};
				return {static_cast<std::byte*>(slice), sizeBytes};
			}
			if (desc.hostAccess == HostAccess::Staged) {
				return beginStaged(desc);
			}
			return {}; // HostAccess::None -- not host-writable
		}

		if (auto tex = GetTexture(resolvedId)) {
			const ResourceDesc& desc = tex->GetDesc();
			if (desc.hostAccess != HostAccess::Staged) {
				return {}; // None, or Mapped (textures have no Mapped counterpart)
			}
			return beginStaged(desc);
		}

		return {};
	}

	inline void PhysicalResourceRegistry::EndHostWrite(ResourceId id, CommandBuffer cmd) {
		const ResourceId resolvedId = ResolveId(id);
		auto             pendingIt = m_pendingHostWrites.find(resolvedId);
		if (pendingIt == m_pendingHostWrites.end()) {
			return;
		}
		const auto [frameIndex, sizeBytes] = pendingIt->second;
		m_pendingHostWrites.erase(pendingIt);

		if (auto buf = GetBuffer(resolvedId)) {
			if (buf->GetDesc().hostAccess == HostAccess::Mapped) {
				buf->FlushSlice(frameIndex);
				return;
			}
			if (buf->GetDesc().hostAccess == HostAccess::Staged && cmd.vkCmd) {
				auto stagingIt = m_stagingBuffers.find(resolvedId);
				if (stagingIt == m_stagingBuffers.end()) {
					return;
				}
				StagingSlot&      slot = stagingIt->second.slots[frameIndex % brassica::FRAME_OVERLAP];
				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
				vk::BufferCopy    region{0, 0, sizeBytes};
				vkCmd.copyBuffer(slot.buffer, buf->GetBuffer(), region);
			}
			return;
		}

		if (auto tex = GetTexture(resolvedId)) {
			if (tex->GetDesc().hostAccess == HostAccess::Staged && cmd.vkCmd) {
				auto stagingIt = m_stagingBuffers.find(resolvedId);
				if (stagingIt == m_stagingBuffers.end()) {
					return;
				}
				StagingSlot&        slot = stagingIt->second.slots[frameIndex % brassica::FRAME_OVERLAP];
				vk::CommandBuffer   vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
				const ResourceDesc& desc = tex->GetDesc();
				vk::BufferImageCopy region{};
				region.setBufferOffset(0)
					.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1})
					.setImageExtent({desc.width, desc.height, std::max(1u, desc.depth)});
				vkCmd.copyBufferToImage(slot.buffer, tex->GetImage(), tex->GetCurrentLayout(), region);
			}
		}
	}

} // namespace brassica::graph
