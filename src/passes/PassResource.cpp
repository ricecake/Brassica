#include "passes/PassResource.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	namespace {
		void TransitionImageLayout(
			vk::CommandBuffer       cmd,
			vk::Image               image,
			vk::ImageLayout         oldLayout,
			vk::ImageLayout         newLayout,
			vk::PipelineStageFlags2 srcStage,
			vk::PipelineStageFlags2 dstStage,
			vk::AccessFlags2        srcAccess,
			vk::AccessFlags2        dstAccess,
			vk::ImageAspectFlags    aspectFlags = vk::ImageAspectFlagBits::eColor
		) {
			vk::ImageMemoryBarrier2 barrier{};
			barrier.setSrcStageMask(srcStage);
			barrier.setSrcAccessMask(srcAccess);
			barrier.setDstStageMask(dstStage);
			barrier.setDstAccessMask(dstAccess);
			barrier.setOldLayout(oldLayout);
			barrier.setNewLayout(newLayout);
			barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setSubresourceRange(vk::ImageSubresourceRange(aspectFlags, 0, 1, 0, 1));
			barrier.setImage(image);

			vk::DependencyInfo depInfo{};
			depInfo.setImageMemoryBarriers(barrier);

			cmd.pipelineBarrier2(depInfo);
		}

		void BufferBarrier(
			vk::CommandBuffer       cmd,
			vk::Buffer              buffer,
			vk::DeviceSize          size,
			vk::PipelineStageFlags2 srcStage,
			vk::PipelineStageFlags2 dstStage,
			vk::AccessFlags2        srcAccess,
			vk::AccessFlags2        dstAccess
		) {
			vk::BufferMemoryBarrier2 barrier{};
			barrier.setSrcStageMask(srcStage);
			barrier.setSrcAccessMask(srcAccess);
			barrier.setDstStageMask(dstStage);
			barrier.setDstAccessMask(dstAccess);
			barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setBuffer(buffer);
			barrier.setOffset(0);
			barrier.setSize(size);

			vk::DependencyInfo depInfo{};
			depInfo.setBufferMemoryBarriers(barrier);

			cmd.pipelineBarrier2(depInfo);
		}
	} // namespace

	// FrameGraphTexture2D implementation
	void FrameGraphTexture2D::create(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent = VkExtent3D{desc.extent.width, desc.extent.height, 1};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = static_cast<VkFormat>(desc.format);
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = static_cast<VkImageUsageFlags>(desc.usage);
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		VkImage vkImg = VK_NULL_HANDLE;
		if (vmaCreateImage(rc.allocator, &imageInfo, &allocInfo, &vkImg, &allocation, nullptr) != VK_SUCCESS) {
			spdlog::error("Failed to create FrameGraphTexture2D image via VMA");
			return;
		}
		image = vkImg;

		vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
		if (desc.format == vk::Format::eD32Sfloat || desc.format == vk::Format::eD24UnormS8Uint || desc.format == vk::Format::eD16Unorm) {
			aspect = vk::ImageAspectFlagBits::eDepth;
		}

		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.setImage(image);
		viewInfo.setViewType(vk::ImageViewType::e2D);
		viewInfo.setFormat(desc.format);
		viewInfo.setSubresourceRange(vk::ImageSubresourceRange(aspect, 0, 1, 0, 1));

		try {
			imageView = rc.device.createImageView(viewInfo);
		} catch (const vk::SystemError& e) {
			spdlog::error("Failed to create FrameGraphTexture2D imageView: {}", e.what());
		}
		currentLayout = vk::ImageLayout::eUndefined;
	}

	void FrameGraphTexture2D::destroy(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		if (imageView) {
			rc.device.destroyImageView(imageView);
			imageView = nullptr;
		}
		if (image && allocation) {
			vmaDestroyImage(rc.allocator, image, allocation);
			image = nullptr;
			allocation = VK_NULL_HANDLE;
		}
	}

	void FrameGraphTexture2D::preRead(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		vk::CommandBuffer cmd = rc->commandBuffer;

		vk::ImageLayout targetLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		if (flags & static_cast<uint32_t>(TextureUsage::StorageRead)) {
			targetLayout = vk::ImageLayout::eGeneral;
		}

		if (currentLayout != targetLayout) {
			TransitionImageLayout(
				cmd,
				image,
				currentLayout,
				targetLayout,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::AccessFlagBits2::eMemoryWrite,
				vk::AccessFlagBits2::eMemoryRead
			);
			currentLayout = targetLayout;
		}
	}

	void FrameGraphTexture2D::preWrite(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		vk::CommandBuffer cmd = rc->commandBuffer;

		vk::ImageLayout targetLayout = vk::ImageLayout::eColorAttachmentOptimal;
		if (flags & static_cast<uint32_t>(TextureUsage::DepthStencilAttachment)) {
			targetLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
		} else if (flags & static_cast<uint32_t>(TextureUsage::StorageWrite)) {
			targetLayout = vk::ImageLayout::eGeneral;
		}

		vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
		if (targetLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
			aspect = vk::ImageAspectFlagBits::eDepth;
		}

		if (currentLayout != targetLayout) {
			TransitionImageLayout(
				cmd,
				image,
				currentLayout,
				targetLayout,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
				vk::AccessFlagBits2::eMemoryWrite,
				aspect
			);
			currentLayout = targetLayout;
		}
	}

	// FrameGraphTexture3D implementation
	void FrameGraphTexture3D::create(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		imageInfo.imageType = VK_IMAGE_TYPE_3D;
		imageInfo.extent = VkExtent3D{desc.extent.width, desc.extent.height, desc.extent.depth};
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = static_cast<VkFormat>(desc.format);
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = static_cast<VkImageUsageFlags>(desc.usage);
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

		VkImage vkImg = VK_NULL_HANDLE;
		if (vmaCreateImage(rc.allocator, &imageInfo, &allocInfo, &vkImg, &allocation, nullptr) != VK_SUCCESS) {
			spdlog::error("Failed to create FrameGraphTexture3D image via VMA");
			return;
		}
		image = vkImg;

		vk::ImageViewCreateInfo viewInfo{};
		viewInfo.setImage(image);
		viewInfo.setViewType(vk::ImageViewType::e3D);
		viewInfo.setFormat(desc.format);
		viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		try {
			imageView = rc.device.createImageView(viewInfo);
		} catch (const vk::SystemError& e) {
			spdlog::error("Failed to create FrameGraphTexture3D imageView: {}", e.what());
		}
		currentLayout = vk::ImageLayout::eUndefined;
	}

	void FrameGraphTexture3D::destroy(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		if (imageView) {
			rc.device.destroyImageView(imageView);
			imageView = nullptr;
		}
		if (image && allocation) {
			vmaDestroyImage(rc.allocator, image, allocation);
			image = nullptr;
			allocation = VK_NULL_HANDLE;
		}
	}

	void FrameGraphTexture3D::preRead(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		vk::CommandBuffer cmd = rc->commandBuffer;

		vk::ImageLayout targetLayout = vk::ImageLayout::eGeneral;
		if (currentLayout != targetLayout) {
			TransitionImageLayout(
				cmd,
				image,
				currentLayout,
				targetLayout,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::AccessFlagBits2::eMemoryWrite,
				vk::AccessFlagBits2::eMemoryRead
			);
			currentLayout = targetLayout;
		}
	}

	void FrameGraphTexture3D::preWrite(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		vk::CommandBuffer cmd = rc->commandBuffer;

		vk::ImageLayout targetLayout = vk::ImageLayout::eGeneral;
		if (currentLayout != targetLayout) {
			TransitionImageLayout(
				cmd,
				image,
				currentLayout,
				targetLayout,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
				vk::AccessFlagBits2::eMemoryWrite
			);
			currentLayout = targetLayout;
		}
	}

	// FrameGraphSSBO implementation
	void FrameGraphSSBO::create(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = desc.size;
		bufferInfo.usage = static_cast<VkBufferUsageFlags>(desc.usage);

		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          vkBuf = VK_NULL_HANDLE;
		VmaAllocationInfo allocResultInfo{};
		if (vmaCreateBuffer(rc.allocator, &bufferInfo, &allocCreateInfo, &vkBuf, &allocation, &allocResultInfo) != VK_SUCCESS) {
			spdlog::error("Failed to create FrameGraphSSBO buffer via VMA");
			return;
		}
		buffer = vkBuf;
		mappedData = allocResultInfo.pMappedData;
	}

	void FrameGraphSSBO::destroy(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		if (buffer && allocation) {
			vmaDestroyBuffer(rc.allocator, buffer, allocation);
			buffer = nullptr;
			allocation = VK_NULL_HANDLE;
			mappedData = nullptr;
		}
	}

	void FrameGraphSSBO::preRead(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);

		vk::PipelineStageFlags2 dstStage = vk::PipelineStageFlagBits2::eAllCommands;
		vk::AccessFlags2        dstAccess = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead;

		if (flags & static_cast<uint32_t>(BufferUsage::Indirect)) {
			dstStage |= vk::PipelineStageFlagBits2::eDrawIndirect;
			dstAccess |= vk::AccessFlagBits2::eIndirectCommandRead;
		}

		BufferBarrier(
			rc->commandBuffer,
			buffer,
			desc.size,
			vk::PipelineStageFlagBits2::eAllCommands,
			dstStage,
			vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
			dstAccess
		);
	}

	void FrameGraphSSBO::preWrite(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		BufferBarrier(
			rc->commandBuffer,
			buffer,
			desc.size,
			vk::PipelineStageFlagBits2::eAllCommands,
			vk::PipelineStageFlagBits2::eAllCommands,
			vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
			vk::AccessFlagBits2::eShaderWrite
		);
	}

	// FrameGraphUBO implementation
	void FrameGraphUBO::create(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bufferInfo.size = desc.size;
		bufferInfo.usage = static_cast<VkBufferUsageFlags>(desc.usage);

		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBuffer          vkBuf = VK_NULL_HANDLE;
		VmaAllocationInfo allocResultInfo{};
		if (vmaCreateBuffer(rc.allocator, &bufferInfo, &allocCreateInfo, &vkBuf, &allocation, &allocResultInfo) != VK_SUCCESS) {
			spdlog::error("Failed to create FrameGraphUBO buffer via VMA");
			return;
		}
		buffer = vkBuf;
		mappedData = allocResultInfo.pMappedData;
	}

	void FrameGraphUBO::destroy(const Desc& desc, void* context) {
		if (!context) return;
		auto& rc = *static_cast<RenderContext*>(context);

		if (buffer && allocation) {
			vmaDestroyBuffer(rc.allocator, buffer, allocation);
			buffer = nullptr;
			allocation = VK_NULL_HANDLE;
			mappedData = nullptr;
		}
	}

	void FrameGraphUBO::preRead(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		BufferBarrier(
			rc->commandBuffer,
			buffer,
			desc.size,
			vk::PipelineStageFlagBits2::eHost,
			vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader,
			vk::AccessFlagBits2::eHostWrite,
			vk::AccessFlagBits2::eUniformRead
		);
	}

	void FrameGraphUBO::preWrite(const Desc& desc, uint32_t flags, void* context) {
		if (!context) return;
		auto* rc = static_cast<RenderContext*>(context);
		BufferBarrier(
			rc->commandBuffer,
			buffer,
			desc.size,
			vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader,
			vk::PipelineStageFlagBits2::eHost,
			vk::AccessFlagBits2::eUniformRead,
			vk::AccessFlagBits2::eHostWrite
		);
	}

} // namespace brassica
