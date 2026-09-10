#pragma once

#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vulkan/vulkan.hpp"

namespace brassica {

	// Semantic aliases for common image roles
	enum class ImageUsagePurpose {
		StandardTexture,     // Read by fragment/compute shaders, populated by CPU or transfer
		DepthBuffer,         // Depth/Stencil attachment (GPU only)
		ColorAttachment,     // Render target for offscreen rendering
		ComputeShaderTarget, // Storage image read/written by a Compute shader
		DataReadback         // GPU-written data to be read back by CPU
	};

	// Semantic aliases for buffer roles
	enum class BufferUsagePurpose {
		VertexInput,    // High-speed GPU memory for raw vertex data
		IndexInput,     // High-speed GPU memory for index arrays
		UniformBlock,   // Constant data frequently updated/read by shaders
		StorageCompute, // Structured buffers for compute or heavy data storage
		StagingTransfer // CPU-visible staging memory
	};

	// Configuration options for textures and images
	struct TextureOptions {
		vk::Format          format = vk::Format::eR8G8B8A8Srgb;
		vk::ImageTiling     tiling = vk::ImageTiling::eOptimal;
		uint32_t            mipLevels = 1;
		uint32_t            depth = 1; // 1 for 2D, >1 for 3D textures
		vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;

		VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
		VmaAllocationCreateFlags memoryFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

		vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor;
		vk::ComponentMapping components = vk::ComponentMapping{};

		// Explicit overrides (if non-zero / non-default)
		vk::ImageUsageFlags      customUsageFlags = vk::ImageUsageFlags(0);
		VmaMemoryUsage           customMemoryUsage = VMA_MEMORY_USAGE_UNKNOWN;
		VmaAllocationCreateFlags customAllocFlags = VmaAllocationCreateFlags(0);
	};

	using ImageConfigOptions = TextureOptions;

	// Staging buffer resource tracked against a timeline semaphore completion value
	struct StagingResource {
		VkBuffer      buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = nullptr;
		uint64_t      completionValue = 0;
	};

	// RAII wrapper for GPU Buffer allocations
	class BufferResource {
	public:
		BufferResource(
			vk::Device                   device,
			VmaAllocator                 allocator,
			size_t                       size,
			BufferUsagePurpose           purpose,
			vk::SharingMode              sharingMode = vk::SharingMode::eExclusive,
			const std::vector<uint32_t>& queueFamilyIndices = {}
		):
			m_device(device), m_allocator(allocator), m_size(size) {
			vk::BufferUsageFlags     usageFlags;
			VmaAllocationCreateFlags allocFlags = 0;

			switch (purpose) {
			case BufferUsagePurpose::VertexInput:
				usageFlags = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;
				break;
			case BufferUsagePurpose::IndexInput:
				usageFlags = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
				break;
			case BufferUsagePurpose::UniformBlock:
				usageFlags = vk::BufferUsageFlagBits::eUniformBuffer;
				allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
				break;
			case BufferUsagePurpose::StorageCompute:
				usageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
				break;
			case BufferUsagePurpose::StagingTransfer:
				usageFlags = vk::BufferUsageFlagBits::eTransferSrc;
				allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
				break;
			}

			vk::BufferCreateInfo bufferInfo(
				{},
				size,
				usageFlags,
				sharingMode,
				static_cast<uint32_t>(queueFamilyIndices.size()),
				queueFamilyIndices.data()
			);

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			allocInfo.flags = allocFlags;

			VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
			VkBuffer           rawBuffer = VK_NULL_HANDLE;
			if (vmaCreateBuffer(m_allocator, &cBufferInfo, &allocInfo, &rawBuffer, &m_allocation, &m_allocInfo) !=
			    VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate buffer resource via VMA!");
			}
			m_buffer = rawBuffer;
		}

		~BufferResource() { Cleanup(); }

		// Move-only semantics
		BufferResource(const BufferResource&) = delete;
		BufferResource& operator=(const BufferResource&) = delete;

		BufferResource(BufferResource&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_buffer(o.m_buffer),
			m_allocation(o.m_allocation),
			m_allocInfo(o.m_allocInfo),
			m_size(o.m_size) {
			o.m_buffer = VK_NULL_HANDLE;
			o.m_allocation = nullptr;
			o.m_allocInfo = {};
			o.m_size = 0;
		}

		BufferResource& operator=(BufferResource&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_buffer = o.m_buffer;
				m_allocation = o.m_allocation;
				m_allocInfo = o.m_allocInfo;
				m_size = o.m_size;

				o.m_buffer = VK_NULL_HANDLE;
				o.m_allocation = nullptr;
				o.m_allocInfo = {};
				o.m_size = 0;
			}
			return *this;
		}

		// Direct access helper for host-mapped memory blocks with proper memory flush when host-coherency is absent
		void UpdateData(const void* srcData, size_t size, size_t offset = 0) {
			if (!m_allocInfo.pMappedData) {
				throw std::runtime_error("Cannot direct-write to an unmapped buffer!");
			}
			if (offset + size > m_size) {
				throw std::out_of_range("Buffer write range exceeds allocated buffer size!");
			}

			std::memcpy(static_cast<char*>(m_allocInfo.pMappedData) + offset, srcData, size);

			// Guarantee visibility if memory is not implicitly host coherent
			VkMemoryPropertyFlags memFlags = 0;
			vmaGetMemoryTypeProperties(m_allocator, m_allocInfo.memoryType, &memFlags);
			if ((memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
				vmaFlushAllocation(m_allocator, m_allocation, offset, size);
			}
		}

		// Getters
		vk::Buffer GetBuffer() const { return m_buffer; }

		VmaAllocation GetAllocation() const { return m_allocation; }

		VmaAllocationInfo GetAllocInfo() const { return m_allocInfo; }

		size_t GetSize() const { return m_size; }

	private:
		void Cleanup() {
			if (m_buffer && m_allocation) {
				vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
				m_buffer = VK_NULL_HANDLE;
				m_allocation = nullptr;
			}
		}

		vk::Device        m_device = nullptr;
		VmaAllocator      m_allocator = nullptr;
		vk::Buffer        m_buffer = VK_NULL_HANDLE;
		VmaAllocation     m_allocation = nullptr;
		VmaAllocationInfo m_allocInfo{};
		size_t            m_size = 0;
	};

	// RAII wrapper for GPU 2D/3D Texture Image resources
	class Texture2D {
	public:
		Texture2D(
			vk::Device                   device,
			VmaAllocator                 allocator,
			uint32_t                     width,
			uint32_t                     height,
			TextureOptions               options = {},
			vk::SharingMode              sharingMode = vk::SharingMode::eExclusive,
			const std::vector<uint32_t>& queueFamilyIndices = {}
		):
			m_device(device),
			m_allocator(allocator),
			m_width(width),
			m_height(height),
			m_depth(options.depth),
			m_mipLevels(options.mipLevels),
			m_format(options.format) {
			InitImage(options, sharingMode, queueFamilyIndices);
		}

		Texture2D(
			vk::Device                   device,
			VmaAllocator                 allocator,
			uint32_t                     width,
			uint32_t                     height,
			ImageUsagePurpose            purpose,
			TextureOptions               options = {},
			vk::SharingMode              sharingMode = vk::SharingMode::eExclusive,
			const std::vector<uint32_t>& queueFamilyIndices = {}
		):
			m_device(device),
			m_allocator(allocator),
			m_width(width),
			m_height(height),
			m_depth(options.depth),
			m_mipLevels(options.mipLevels),
			m_format(options.format) {
			vk::ImageUsageFlags      usageFlags;
			VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
			VmaAllocationCreateFlags allocFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
			vk::ImageAspectFlags     aspectFlags = vk::ImageAspectFlagBits::eColor;

			switch (purpose) {
			case ImageUsagePurpose::StandardTexture:
				usageFlags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
				break;
			case ImageUsagePurpose::DepthBuffer:
				usageFlags = vk::ImageUsageFlagBits::eDepthStencilAttachment;
				aspectFlags = vk::ImageAspectFlagBits::eDepth;
				if (m_format == vk::Format::eR8G8B8A8Srgb) {
					m_format = vk::Format::eD32Sfloat;
				}
				break;
			case ImageUsagePurpose::ColorAttachment:
				usageFlags = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
				break;
			case ImageUsagePurpose::ComputeShaderTarget:
				usageFlags = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled;
				break;
			case ImageUsagePurpose::DataReadback:
				usageFlags = vk::ImageUsageFlagBits::eTransferDst;
				options.tiling = vk::ImageTiling::eLinear;
				allocFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
				break;
			}

			if (options.customUsageFlags)
				usageFlags = options.customUsageFlags;
			if (options.customMemoryUsage != VMA_MEMORY_USAGE_UNKNOWN)
				memoryUsage = options.customMemoryUsage;
			if (options.customAllocFlags)
				allocFlags = options.customAllocFlags;

			options.usage = usageFlags;
			options.memoryUsage = memoryUsage;
			options.memoryFlags = allocFlags;
			options.aspectFlags = aspectFlags;

			InitImage(options, sharingMode, queueFamilyIndices);
		}

		~Texture2D() { Cleanup(); }

		// Move-only semantics
		Texture2D(const Texture2D&) = delete;
		Texture2D& operator=(const Texture2D&) = delete;

		Texture2D(Texture2D&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_image(o.m_image),
			m_allocation(o.m_allocation),
			m_view(o.m_view),
			m_sampler(o.m_sampler),
			m_width(o.m_width),
			m_height(o.m_height),
			m_depth(o.m_depth),
			m_mipLevels(o.m_mipLevels),
			m_format(o.m_format),
			m_stagingQueue(std::move(o.m_stagingQueue)) {
			o.m_image = VK_NULL_HANDLE;
			o.m_allocation = nullptr;
			o.m_view = nullptr;
			o.m_sampler = nullptr;
		}

		Texture2D& operator=(Texture2D&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_image = o.m_image;
				m_allocation = o.m_allocation;
				m_view = o.m_view;
				m_sampler = o.m_sampler;
				m_width = o.m_width;
				m_height = o.m_height;
				m_depth = o.m_depth;
				m_mipLevels = o.m_mipLevels;
				m_format = o.m_format;
				m_stagingQueue = std::move(o.m_stagingQueue);

				o.m_image = VK_NULL_HANDLE;
				o.m_allocation = nullptr;
				o.m_view = nullptr;
				o.m_sampler = nullptr;
			}
			return *this;
		}

		// Vulkan 1.3 Synchronization2 pipeline data upload with Timeline Semaphore staging lifecycle management
		void UploadPixelData(
			vk::CommandBuffer cmdBuffer,
			VmaAllocator      allocator,
			const void*       pixelData,
			size_t            dataSize,
			uint64_t          completionValue = 0,
			uint32_t          srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			uint32_t          dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
		) {
			// A. Allocate host-visible staging buffer via VMA
			vk::BufferCreateInfo
				bufferInfo({}, dataSize, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive);
			VmaAllocationCreateInfo stagingAllocInfo{};
			stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
			stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
				VMA_ALLOCATION_CREATE_MAPPED_BIT;

			VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
			VkBuffer           rawStagingBuffer = VK_NULL_HANDLE;
			VmaAllocation      stagingAllocation = nullptr;
			VmaAllocationInfo  stagingAllocData{};

			if (vmaCreateBuffer(
					allocator,
					&cBufferInfo,
					&stagingAllocInfo,
					&rawStagingBuffer,
					&stagingAllocation,
					&stagingAllocData
				) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate staging buffer for texture upload!");
			}

			// B. Copy pixel array into mapped host memory with coherency flush if required
			std::memcpy(stagingAllocData.pMappedData, pixelData, dataSize);

			VkMemoryPropertyFlags memFlags = 0;
			vmaGetMemoryTypeProperties(allocator, stagingAllocData.memoryType, &memFlags);
			if ((memFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
				vmaFlushAllocation(allocator, stagingAllocation, 0, dataSize);
			}

			// C. Record layout barrier using Vulkan 1.3 Synchronization2: Undefined -> Transfer Dst Optimal
			vk::ImageMemoryBarrier2 barrierToTransfer(
				vk::PipelineStageFlagBits2::eTopOfPipe,
				vk::AccessFlagBits2::eNone,
				vk::PipelineStageFlagBits2::eTransfer,
				vk::AccessFlagBits2::eTransferWrite,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal,
				VK_QUEUE_FAMILY_IGNORED,
				VK_QUEUE_FAMILY_IGNORED,
				m_image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, m_mipLevels, 0, 1)
			);

			vk::DependencyInfo depToTransfer({}, {}, {}, barrierToTransfer);
			cmdBuffer.pipelineBarrier2(depToTransfer);

			// D. Record Buffer-to-Image copy
			vk::BufferImageCopy copyRegion(
				0,
				0,
				0,
				vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
				vk::Offset3D(0, 0, 0),
				vk::Extent3D(m_width, m_height, m_depth)
			);

			cmdBuffer
				.copyBufferToImage(rawStagingBuffer, m_image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

			// E. Record layout barrier or Release barrier using Vulkan 1.3 Synchronization2
			bool isQueueTransfer = (srcQueueFamilyIndex != dstQueueFamilyIndex) &&
				(srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) && (dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED);

			vk::PipelineStageFlags2 dstStage = vk::PipelineStageFlagBits2::eFragmentShader |
				vk::PipelineStageFlagBits2::eComputeShader;
			vk::AccessFlags2 dstAccess = vk::AccessFlagBits2::eShaderRead;

			vk::ImageMemoryBarrier2 barrierToShader(
				vk::PipelineStageFlagBits2::eTransfer,
				vk::AccessFlagBits2::eTransferWrite,
				isQueueTransfer ? vk::PipelineStageFlagBits2::eNone : dstStage,
				isQueueTransfer ? vk::AccessFlagBits2::eNone : dstAccess,
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				srcQueueFamilyIndex,
				dstQueueFamilyIndex,
				m_image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, m_mipLevels, 0, 1)
			);

			vk::DependencyInfo depToShader({}, {}, {}, barrierToShader);
			cmdBuffer.pipelineBarrier2(depToShader);

			// F. Enqueue staging resource tied to timeline semaphore value
			m_stagingQueue.push_back(StagingResource{rawStagingBuffer, stagingAllocation, completionValue});
		}

		// Records the matching Acquire barrier on the destination queue command buffer during a queue ownership
		// transfer
		void RecordAcquireBarrier(
			vk::CommandBuffer dstCmdBuffer,
			uint32_t          srcQueueFamilyIndex,
			uint32_t          dstQueueFamilyIndex
		) {
			vk::ImageMemoryBarrier2 acquireBarrier(
				vk::PipelineStageFlagBits2::eNone,
				vk::AccessFlagBits2::eNone,
				vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader,
				vk::AccessFlagBits2::eShaderRead,
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				srcQueueFamilyIndex,
				dstQueueFamilyIndex,
				m_image,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, m_mipLevels, 0, 1)
			);

			vk::DependencyInfo depInfo({}, {}, {}, acquireBarrier);
			dstCmdBuffer.pipelineBarrier2(depInfo);
		}

		// Cleans up staging resources whose completion values have been reached on GPU timeline semaphore
		void CleanupStagingResources(uint64_t currentTimelineValue) {
			while (!m_stagingQueue.empty()) {
				if (currentTimelineValue >= m_stagingQueue.front().completionValue) {
					vmaDestroyBuffer(m_allocator, m_stagingQueue.front().buffer, m_stagingQueue.front().allocation);
					m_stagingQueue.pop_front();
				} else {
					break;
				}
			}
		}

		// Force cleanup of all active staging resources
		void CleanupStagingResources() {
			for (auto& res : m_stagingQueue) {
				if (res.buffer && res.allocation) {
					vmaDestroyBuffer(m_allocator, res.buffer, res.allocation);
				}
			}
			m_stagingQueue.clear();
		}

		// Creates default texture sampler
		void CreateDefaultSampler(vk::Filter filter = vk::Filter::eLinear) {
			if (m_sampler) {
				m_device.destroySampler(m_sampler);
				m_sampler = nullptr;
			}
			vk::SamplerCreateInfo samplerInfo(
				{},
				filter,
				filter,
				vk::SamplerMipmapMode::eLinear,
				vk::SamplerAddressMode::eRepeat,
				vk::SamplerAddressMode::eRepeat,
				vk::SamplerAddressMode::eRepeat,
				0.0f,
				VK_FALSE,
				1.0f,
				VK_FALSE,
				vk::CompareOp::eAlways,
				0.0f,
				static_cast<float>(m_mipLevels),
				vk::BorderColor::eIntOpaqueBlack,
				VK_FALSE
			);
			m_sampler = m_device.createSampler(samplerInfo);
		}

		// Binds texture view & sampler into target descriptor set
		void BindToDescriptorSet(vk::DescriptorSet targetSet, uint32_t bindingIndex = 0) {
			if (!m_sampler) {
				CreateDefaultSampler();
			}

			vk::DescriptorImageInfo imageInfo(m_sampler, m_view, vk::ImageLayout::eShaderReadOnlyOptimal);

			vk::WriteDescriptorSet writeInfo(
				targetSet,
				bindingIndex,
				0,
				1,
				vk::DescriptorType::eCombinedImageSampler,
				&imageInfo,
				nullptr,
				nullptr
			);

			m_device.updateDescriptorSets(1, &writeInfo, 0, nullptr);
		}

		// Getters
		vk::Image GetImage() const { return m_image; }

		vk::ImageView GetImageView() const { return m_view; }

		vk::Sampler GetSampler() const { return m_sampler; }

		VmaAllocation GetAllocation() const { return m_allocation; }

		uint32_t GetWidth() const { return m_width; }

		uint32_t GetHeight() const { return m_height; }

		uint32_t GetDepth() const { return m_depth; }

		vk::Format GetFormat() const { return m_format; }

	private:
		void InitImage(
			const TextureOptions&        options,
			vk::SharingMode              sharingMode,
			const std::vector<uint32_t>& queueFamilyIndices
		) {
			vk::ImageType     imageType = (m_depth > 1) ? vk::ImageType::e3D : vk::ImageType::e2D;
			vk::ImageViewType viewType = (m_depth > 1) ? vk::ImageViewType::e3D : vk::ImageViewType::e2D;

			vk::ImageCreateInfo imageInfo(
				{},
				imageType,
				m_format,
				vk::Extent3D(m_width, m_height, m_depth),
				options.mipLevels,
				1,
				vk::SampleCountFlagBits::e1,
				options.tiling,
				options.usage,
				sharingMode,
				static_cast<uint32_t>(queueFamilyIndices.size()),
				queueFamilyIndices.data()
			);

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = options.memoryUsage;
			allocInfo.flags = options.memoryFlags;

			VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
			VkImage           rawImage = VK_NULL_HANDLE;
			if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate texture image via VMA!");
			}
			m_image = rawImage;

			vk::ImageViewCreateInfo viewInfo(
				{},
				m_image,
				viewType,
				m_format,
				options.components,
				vk::ImageSubresourceRange(options.aspectFlags, 0, options.mipLevels, 0, 1)
			);
			m_view = m_device.createImageView(viewInfo);
		}

		void Cleanup() {
			// If in-flight transfers exist on destruction, wait for GPU completion to prevent race conditions/crashes
			if (!m_stagingQueue.empty() && m_device) {
				m_device.waitIdle();
			}
			CleanupStagingResources();
			if (m_sampler) {
				m_device.destroySampler(m_sampler);
				m_sampler = nullptr;
			}
			if (m_view) {
				m_device.destroyImageView(m_view);
				m_view = nullptr;
			}
			if (m_image && m_allocation) {
				vmaDestroyImage(m_allocator, m_image, m_allocation);
				m_image = VK_NULL_HANDLE;
				m_allocation = nullptr;
			}
		}

		vk::Device                  m_device = nullptr;
		VmaAllocator                m_allocator = nullptr;
		vk::Image                   m_image = VK_NULL_HANDLE;
		VmaAllocation               m_allocation = nullptr;
		vk::ImageView               m_view = nullptr;
		vk::Sampler                 m_sampler = nullptr;
		uint32_t                    m_width = 0, m_height = 0, m_depth = 1;
		uint32_t                    m_mipLevels = 1;
		vk::Format                  m_format = vk::Format::eR8G8B8A8Srgb;
		std::deque<StagingResource> m_stagingQueue;
	};

	using TextureResource = Texture2D;

	// Scalable Descriptor Pool supporting bindless and heavy asset loads
	class GlobalDescriptorPool {
	public:
		struct CapacityOptions {
			uint32_t maxSets = 4096;
			uint32_t combinedImageSamplers = 4096;
			uint32_t storageBuffers = 2048;
			uint32_t uniformBuffers = 2048;
			uint32_t storageImages = 1024;
			bool     allowUpdateAfterBind = true;
		};

		GlobalDescriptorPool(vk::Device device, CapacityOptions opts = {}): m_device(device) {
			std::vector<vk::DescriptorPoolSize> poolSizes = {
				{vk::DescriptorType::eCombinedImageSampler, opts.combinedImageSamplers},
				{vk::DescriptorType::eStorageBuffer, opts.storageBuffers},
				{vk::DescriptorType::eUniformBuffer, opts.uniformBuffers},
				{vk::DescriptorType::eStorageImage, opts.storageImages}
			};

			vk::DescriptorPoolCreateFlags flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
			if (opts.allowUpdateAfterBind) {
				flags |= vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
			}

			vk::DescriptorPoolCreateInfo
				poolInfo(flags, opts.maxSets, static_cast<uint32_t>(poolSizes.size()), poolSizes.data());
			m_pool = m_device.createDescriptorPool(poolInfo);
		}

		~GlobalDescriptorPool() { Cleanup(); }

		GlobalDescriptorPool(const GlobalDescriptorPool&) = delete;
		GlobalDescriptorPool& operator=(const GlobalDescriptorPool&) = delete;

		GlobalDescriptorPool(GlobalDescriptorPool&& o) noexcept: m_device(o.m_device), m_pool(o.m_pool) {
			o.m_pool = nullptr;
		}

		GlobalDescriptorPool& operator=(GlobalDescriptorPool&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_pool = o.m_pool;
				o.m_pool = nullptr;
			}
			return *this;
		}

		vk::DescriptorSet AllocateSet(vk::DescriptorSetLayout layout) {
			vk::DescriptorSetAllocateInfo allocInfo(m_pool, 1, &layout);
			vk::DescriptorSet             set;
			if (m_device.allocateDescriptorSets(&allocInfo, &set) != vk::Result::eSuccess) {
				throw std::runtime_error("Failed to allocate descriptor set from GlobalDescriptorPool!");
			}
			return set;
		}

		vk::DescriptorPool GetPool() const { return m_pool; }

	private:
		void Cleanup() {
			if (m_pool) {
				m_device.destroyDescriptorPool(m_pool);
				m_pool = nullptr;
			}
		}

		vk::Device         m_device = nullptr;
		vk::DescriptorPool m_pool = nullptr;
	};

	struct DescriptorLayoutOptions {
		uint32_t                   bindingIndex = 0;
		vk::DescriptorType         type = vk::DescriptorType::eCombinedImageSampler;
		uint32_t                   count = 1;
		vk::ShaderStageFlags       stageFlags = vk::ShaderStageFlagBits::eFragment;
		vk::DescriptorBindingFlags bindingFlags = {};
	};

	// Manages descriptor set layouts and allocations from a pool
	class DescriptorSetManager {
	public:
		DescriptorSetManager(
			vk::Device              device,
			DescriptorLayoutOptions options = {},
			GlobalDescriptorPool*   globalPool = nullptr
		):
			m_device(device), m_globalPool(globalPool) {
			vk::DescriptorSetLayoutBinding
				layoutBinding(options.bindingIndex, options.type, options.count, options.stageFlags, nullptr);

			vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo(1, &options.bindingFlags);

			vk::DescriptorSetLayoutCreateFlags layoutFlags{};
			if ((options.bindingFlags & vk::DescriptorBindingFlagBits::eUpdateAfterBind) ==
			    vk::DescriptorBindingFlagBits::eUpdateAfterBind) {
				layoutFlags |= vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
			}

			vk::DescriptorSetLayoutCreateInfo layoutInfo(layoutFlags, 1, &layoutBinding);
			if (options.bindingFlags != vk::DescriptorBindingFlags{}) {
				layoutInfo.pNext = &bindingFlagsInfo;
			}

			m_layout = m_device.createDescriptorSetLayout(layoutInfo);

			if (!m_globalPool) {
				// Internal pool with scaled capacity for multiple descriptor sets
				vk::DescriptorPoolSize        poolSize(options.type, options.count * 128);
				vk::DescriptorPoolCreateFlags poolFlags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
				if ((options.bindingFlags & vk::DescriptorBindingFlagBits::eUpdateAfterBind) ==
				    vk::DescriptorBindingFlagBits::eUpdateAfterBind) {
					poolFlags |= vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
				}

				vk::DescriptorPoolCreateInfo poolInfo(poolFlags, 128, 1, &poolSize);
				m_internalPool = m_device.createDescriptorPool(poolInfo);
			}
		}

		~DescriptorSetManager() { Cleanup(); }

		DescriptorSetManager(const DescriptorSetManager&) = delete;
		DescriptorSetManager& operator=(const DescriptorSetManager&) = delete;

		DescriptorSetManager(DescriptorSetManager&& o) noexcept:
			m_device(o.m_device), m_layout(o.m_layout), m_internalPool(o.m_internalPool), m_globalPool(o.m_globalPool) {
			o.m_layout = nullptr;
			o.m_internalPool = nullptr;
		}

		DescriptorSetManager& operator=(DescriptorSetManager&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_layout = o.m_layout;
				m_internalPool = o.m_internalPool;
				m_globalPool = o.m_globalPool;

				o.m_layout = nullptr;
				o.m_internalPool = nullptr;
			}
			return *this;
		}

		vk::DescriptorSet AllocateSet() {
			if (m_globalPool) {
				return m_globalPool->AllocateSet(m_layout);
			}

			vk::DescriptorSetAllocateInfo allocInfo(m_internalPool, 1, &m_layout);
			vk::DescriptorSet             set;
			if (m_device.allocateDescriptorSets(&allocInfo, &set) != vk::Result::eSuccess) {
				throw std::runtime_error("Failed to allocate descriptor set from DescriptorSetManager!");
			}
			return set;
		}

		vk::DescriptorSetLayout GetLayout() const { return m_layout; }

	private:
		void Cleanup() {
			if (m_internalPool) {
				m_device.destroyDescriptorPool(m_internalPool);
				m_internalPool = nullptr;
			}
			if (m_layout) {
				m_device.destroyDescriptorSetLayout(m_layout);
				m_layout = nullptr;
			}
		}

		vk::Device              m_device = nullptr;
		vk::DescriptorSetLayout m_layout = nullptr;
		vk::DescriptorPool      m_internalPool = nullptr;
		GlobalDescriptorPool*   m_globalPool = nullptr;
	};

	// Asset Manager consolidating texture and buffer resource lifecycle
	class AssetManager {
	public:
		AssetManager(vk::Device device, VmaAllocator allocator): m_device(device), m_allocator(allocator) {}

		~AssetManager() { ClearAllResources(); }

		AssetManager(const AssetManager&) = delete;
		AssetManager& operator=(const AssetManager&) = delete;

		AssetManager(AssetManager&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_textures(std::move(o.m_textures)),
			m_buffers(std::move(o.m_buffers)) {}

		AssetManager& operator=(AssetManager&& o) noexcept {
			if (this != &o) {
				ClearAllResources();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_textures = std::move(o.m_textures);
				m_buffers = std::move(o.m_buffers);
			}
			return *this;
		}

		// --- TEXTURE FACTORIES ---

		Texture2D*
		CreateTexture(const std::string& name, uint32_t width, uint32_t height, TextureOptions options = {}) {
			auto it = m_textures.find(name);
			if (it != m_textures.end()) {
				std::cout << "[Warning] Texture '" << name << "' already exists. Returning existing resource.\n";
				return &it->second;
			}

			auto [emplaceIt, success] = m_textures.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(name),
				std::forward_as_tuple(m_device, m_allocator, width, height, options)
			);
			return &emplaceIt->second;
		}

		Texture2D* Create2DTexture(const std::string& name, uint32_t width, uint32_t height, TextureOptions opts = {}) {
			return CreateTexture(name, width, height, opts);
		}

		Texture2D* Create3DTexture(
			const std::string& name,
			uint32_t           width,
			uint32_t           height,
			uint32_t           depth,
			TextureOptions     opts = {}
		) {
			opts.depth = depth;
			return CreateTexture(name, width, height, opts);
		}

		Texture2D*
		CreateDepthBuffer(const std::string& name, uint32_t width, uint32_t height, TextureOptions opts = {}) {
			auto it = m_textures.find(name);
			if (it != m_textures.end())
				return &it->second;

			auto [emplaceIt, success] = m_textures.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(name),
				std::forward_as_tuple(m_device, m_allocator, width, height, ImageUsagePurpose::DepthBuffer, opts)
			);
			return &emplaceIt->second;
		}

		Texture2D*
		CreateComputeShaderTarget(const std::string& name, uint32_t width, uint32_t height, TextureOptions opts = {}) {
			auto it = m_textures.find(name);
			if (it != m_textures.end())
				return &it->second;

			auto [emplaceIt, success] = m_textures.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(name),
				std::forward_as_tuple(
					m_device,
					m_allocator,
					width,
					height,
					ImageUsagePurpose::ComputeShaderTarget,
					opts
				)
			);
			return &emplaceIt->second;
		}

		// --- BUFFER FACTORIES ---

		BufferResource* CreateVertexBuffer(const std::string& name, size_t size) {
			auto it = m_buffers.find(name);
			if (it != m_buffers.end())
				return &it->second;

			auto [emplaceIt, success] = m_buffers.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(name),
				std::forward_as_tuple(m_device, m_allocator, size, BufferUsagePurpose::VertexInput)
			);
			return &emplaceIt->second;
		}

		BufferResource* CreateUniformBuffer(const std::string& name, size_t size) {
			auto it = m_buffers.find(name);
			if (it != m_buffers.end())
				return &it->second;

			auto [emplaceIt, success] = m_buffers.emplace(
				std::piecewise_construct,
				std::forward_as_tuple(name),
				std::forward_as_tuple(m_device, m_allocator, size, BufferUsagePurpose::UniformBlock)
			);
			return &emplaceIt->second;
		}

		// Resource Lookup
		Texture2D* GetTexture(const std::string& name) {
			auto it = m_textures.find(name);
			if (it != m_textures.end())
				return &it->second;
			return nullptr;
		}

		BufferResource* GetBuffer(const std::string& name) {
			auto it = m_buffers.find(name);
			if (it != m_buffers.end())
				return &it->second;
			return nullptr;
		}

		void UnloadTexture(const std::string& name) { m_textures.erase(name); }

		void UnloadBuffer(const std::string& name) { m_buffers.erase(name); }

		void CleanupStagingResources(uint64_t currentTimelineValue) {
			for (auto& [name, tex] : m_textures) {
				tex.CleanupStagingResources(currentTimelineValue);
			}
		}

		void ClearAllTextures() { m_textures.clear(); }

		void ClearAllBuffers() { m_buffers.clear(); }

		void ClearAllResources() {
			m_textures.clear();
			m_buffers.clear();
		}

	private:
		vk::Device                                      m_device = nullptr;
		VmaAllocator                                    m_allocator = nullptr;
		std::unordered_map<std::string, Texture2D>      m_textures;
		std::unordered_map<std::string, BufferResource> m_buffers;
	};

} // namespace brassica
