#include <vk_mem_alloc.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vulkan/vulkan.hpp"

struct TextureOptions {
	vk::Format          format = vk::Format::eR8G8B8A8Srgb;
	vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageTiling     tiling = vk::ImageTiling::eOptimal;
	uint32_t            mipLevels = 1;

	VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationCreateFlags memoryFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

	vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor;
	vk::ComponentMapping components = vk::ComponentMapping{};
};

class Texture2D {
public:
	Texture2D(vk::Device device, VmaAllocator allocator, uint32_t width, uint32_t height, TextureOptions options = {}):
		m_device(device), m_allocator(allocator), m_width(width), m_height(height), m_format(options.format) {
		// 1. Build and create the Vulkan Image with VMA memory allocation
		vk::ImageCreateInfo imageInfo(
			{},
			vk::ImageType::e2D,
			m_format,
			vk::Extent3D(width, height, 1),
			options.mipLevels,
			1,
			vk::SampleCountFlagBits::e1,
			options.tiling,
			options.usage,
			vk::SharingMode::eExclusive
		);

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = options.memoryUsage;
		allocInfo.flags = options.memoryFlags;

		VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
		VkImage           rawImage;
		if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate texture image via VMA!");
		}
		m_image = rawImage;

		// 2. Automatically generate the companion Image View
		vk::ImageViewCreateInfo viewInfo(
			{},
			m_image,
			vk::ImageViewType::e2D,
			m_format,
			options.components,
			vk::ImageSubresourceRange(options.aspectFlags, 0, options.mipLevels, 0, 1)
		);

		m_view = m_device.createImageView(viewInfo);
	}

	~Texture2D() {
		if (m_view)
			m_device.destroyImageView(m_view);
		if (m_image && m_allocation)
			vmaDestroyImage(m_allocator, m_image, m_allocation);
		if (m_sampler)
			CleanupSampler();
	}

	// Move-only operations
	Texture2D(const Texture2D&) = delete;
	Texture2D& operator=(const Texture2D&) = delete;

	Texture2D(Texture2D&& other) noexcept:
		m_device(other.m_device),
		m_allocator(other.m_allocator),
		m_image(other.m_image),
		m_allocation(other.m_allocation),
		m_view(other.m_view),
		m_width(other.m_width),
		m_height(other.m_height),
		m_format(other.m_format) {
		other.m_image = nullptr;
		other.m_allocation = nullptr;
		other.m_view = nullptr;
	}

	// Pipeline Data Upload Function
	void UploadPixelData(vk::CommandBuffer cmdBuffer, VmaAllocator allocator, const void* pixelData, size_t dataSize) {
		// A. Allocate a temporary, host-visible staging buffer via VMA
		vk::BufferCreateInfo
			bufferInfo({}, dataSize, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive);
		VmaAllocationCreateInfo stagingAllocInfo{};
		stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
		VkBuffer           rawStagingBuffer;
		VmaAllocation      stagingAllocation;
		VmaAllocationInfo  stagingAllocData;

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

		// B. Memory copy your raw pixel array into the mapped CPU pointer
		std::memcpy(stagingAllocData.pMappedData, pixelData, dataSize);

		// C. Record layout barrier: Undefined -> Transfer Dst Optimal
		vk::ImageMemoryBarrier barrierToTransfer(
			vk::AccessFlagBits::eNone,
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			m_image,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		);

		cmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			{},
			nullptr,
			nullptr,
			barrierToTransfer
		);

		// D. Record the Buffer-to-Image copy transaction
		vk::BufferImageCopy copyRegion(
			0,
			0,
			0,
			vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
			vk::Offset3D(0, 0, 0),
			vk::Extent3D(m_width, m_height, 1)
		);

		cmdBuffer.copyBufferToImage(rawStagingBuffer, m_image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

		// E. Record layout barrier: Transfer Dst Optimal -> Shader Read Only Optimal
		vk::ImageMemoryBarrier barrierToShader(
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			m_image,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		);

		cmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{},
			nullptr,
			nullptr,
			barrierToShader
		);

		// F. Schedule clean up of the staging buffer after the GPU pipeline processing completes
		// Note: In an engine loop, you must defer 'vmaDestroyBuffer' until after this frame's fence is signaled!
		m_deferredStagingBuffers.push_back({rawStagingBuffer, stagingAllocation});
	}

	// Generates a boilerplate texture sampler with sensible filtering defaults
	void CreateDefaultSampler(vk::Filter filter = vk::Filter::eLinear) {
		vk::SamplerCreateInfo samplerInfo(
			{},
			filter,
			filter, // Mag and Min filters
			vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			0.0f,
			VK_FALSE,
			1.0f, // Anisotropy disabled by default
			VK_FALSE,
			vk::CompareOp::eAlways,
			0.0f,
			0.0f,
			vk::BorderColor::eIntOpaqueBlack,
			VK_FALSE
		);
		m_sampler = m_device.createSampler(samplerInfo);
	}

	// Explicitly write our image resources into a target Descriptor Set
	void BindToDescriptorSet(vk::DescriptorSet targetSet, uint32_t bindingIndex = 0) {
		if (!m_sampler) {
			CreateDefaultSampler(); // Guarantee we have a sampler layout active
		}

		// Struct containing the concrete texture asset bindings
		vk::DescriptorImageInfo imageInfo(
			m_sampler,
			m_view,
			vk::ImageLayout::eShaderReadOnlyOptimal // The layout we transitioned to during upload
		);

		// Record the layout update instruction
		vk::WriteDescriptorSet writeInfo(
			targetSet,
			bindingIndex,
			0, // Array element offset
			1, // Descriptor count
			vk::DescriptorType::eCombinedImageSampler,
			&imageInfo,
			nullptr,
			nullptr
		);

		m_device.updateDescriptorSets(1, &writeInfo, 0, nullptr);
	}

	// Clean up inside destructor
	void CleanupSampler() {
		if (m_sampler) {
			m_device.destroySampler(m_sampler);
			m_sampler = nullptr;
		}
	}

	vk::Sampler GetSampler() const { return m_sampler; }

	void CleanupStagingResources() {
		for (auto& res : m_deferredStagingBuffers) {
			vmaDestroyBuffer(m_allocator, res.first, res.second);
		}
		m_deferredStagingBuffers.clear();
	}

	// Getters
	vk::Image GetImage() const { return m_image; }

	vk::ImageView GetImageView() const { return m_view; }

private:
	vk::Device    m_device;
	VmaAllocator  m_allocator;
	vk::Image     m_image;
	VmaAllocation m_allocation;
	vk::ImageView m_view;
	uint32_t      m_width;
	uint32_t      m_height;
	vk::Format    m_format;
	vk::Sampler   m_sampler = nullptr;

	std::vector<std::pair<VkBuffer, VmaAllocation>> m_deferredStagingBuffers;
};

struct DescriptorLayoutOptions {
	uint32_t             bindingIndex = 0;
	vk::DescriptorType   type = vk::DescriptorType::eCombinedImageSampler;
	uint32_t             count = 1;
	vk::ShaderStageFlags stageFlags = vk::ShaderStageFlagBits::eFragment;
};

class DescriptorSetManager {
public:
	DescriptorSetManager(vk::Device device, DescriptorLayoutOptions options = {}): m_device(device) {
		// 1. Define the structural binding configuration
		vk::DescriptorSetLayoutBinding
			layoutBinding(options.bindingIndex, options.type, options.count, options.stageFlags, nullptr);

		vk::DescriptorSetLayoutCreateInfo layoutInfo({}, 1, &layoutBinding);
		m_layout = m_device.createDescriptorSetLayout(layoutInfo);

		// 2. Build a descriptor pool sized specifically for allocating one set
		vk::DescriptorPoolSize       poolSize(options.type, options.count);
		vk::DescriptorPoolCreateInfo poolInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, // Allows resetting individual sets
			1,                                                    // Max sets from this pool
			1,
			&poolSize
		);
		m_pool = m_device.createDescriptorPool(poolInfo);
	}

	~DescriptorSetManager() {
		if (m_pool)
			m_device.destroyDescriptorPool(m_pool);
		if (m_layout)
			m_device.destroyDescriptorSetLayout(m_layout);
	}

	// Allocate an empty Descriptor Set using our layout
	vk::DescriptorSet AllocateSet() {
		vk::DescriptorSetAllocateInfo allocInfo(m_pool, 1, &m_layout);

		vk::DescriptorSet descriptorSet;
		if (m_device.allocateDescriptorSets(&allocInfo, &descriptorSet) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to allocate descriptor set!");
		}
		return descriptorSet;
	}

	// Getters
	vk::DescriptorSetLayout GetLayout() const { return m_layout; }

private:
	vk::Device              m_device;
	vk::DescriptorSetLayout m_layout;
	vk::DescriptorPool      m_pool;
};

class AssetManager {
public:
	AssetManager(vk::Device device, VmaAllocator allocator): m_device(device), m_allocator(allocator) {}

	// Destructor guarantees everything is destroyed in order before allocator closes
	~AssetManager() { ClearAllTextures(); }

	// Disable copying to enforce strict single-ownership of GPU assets
	AssetManager(const AssetManager&) = delete;
	AssetManager& operator=(const AssetManager&) = delete;

	/**
	 * @brief Creates a texture using defaults or custom options, and tracks it.
	 * @return A raw pointer to the managed texture.
	 */
	Texture2D* CreateTexture(const std::string& name, uint32_t width, uint32_t height, TextureOptions options = {}) {
		// Prevent duplicate names from overwriting allocations silently
		if (m_textures.find(name) != m_textures.end()) {
			std::cout << "[Warning] Texture with name '" << name << "' already exists. Returning existing resource.\n";
			return &m_textures.at(name);
		}

		// Emplace uses the move constructor to transfer ownership directly into the map
		auto [it, success] = m_textures.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, width, height, options)
		);

		return &it->second;
	}

	/**
	 * @brief Retrieves a pointer to a managed texture by its lookup string.
	 */
	Texture2D* GetTexture(const std::string& name) {
		auto it = m_textures.find(name);
		if (it != m_textures.end()) {
			return &it->second;
		}
		std::cerr << "[Error] Texture '" << name << "' not found in Asset Manager.\n";
		return nullptr;
	}

	/**
	 * @brief Manually releases a specific asset when it's no longer needed (e.g., leaving a level).
	 */
	void UnloadTexture(const std::string& name) {
		auto it = m_textures.find(name);
		if (it != m_textures.end()) {
			// Because our destructor triggers the internal Vulkan/VMA destroy code,
			// erasing from the map automatically cleans up the GPU handles!
			m_textures.erase(it);
		}
	}

	/**
	 * @brief Drops all active allocations cleanly. Call this on engine shutdown.
	 */
	void ClearAllTextures() { m_textures.clear(); }

private:
	vk::Device   m_device;
	VmaAllocator m_allocator;

	// Maps a readable string descriptor to our move-only wrapper block
	std::unordered_map<std::string, Texture2D> m_textures;
};

#pragma once
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace Brassica {
	/*
	    The basic notion is to have phases and graphs both be able to act like nodes, so that
	    the graph can hold a group of them, and sort the phases in order, and nodes can specify
	    that they come before or after different phases, and graphs can be treated as nodes so
	    that complex nodes can themselves be comprised of nodes.
	*/
	enum class ExecutionDomain { Graphics, Compute, Host, Transfer };

	struct MemoryBarrier {};

	struct CommandBuffer {};

	struct RenderContext {}; // Carries frame state (e.g., UI toggles, camera data)

	template <typename T>
	concept ResourceKey = std::is_empty_v<T> && std::is_trivially_constructible_v<T>;
	template <typename T>
	concept NodeLike = requires() { true; };

	template <ResourceKey T>
	struct History {
		using Type = T;
	};

	template <typename T>
	concept NodeLike = requires(T t, const RenderContext& ctx, CommandBuffer& cmd) {
		{ t.Setup(ctx) } -> std::same_as<Recipe>;
		{ t.Execute(cmd) };
	};

	class Phase {};

	class Node {};

	class Graph {
	public:
		using ExpectedBuffers = std::tuple<>;
	};

	class Queue {};

	template <typename T>
	class Resource {
		void GetReadBarrier() {};
		void GetReleaseBarrier(Queue destQueue) {};
		void GetAcquireBarrier(Queue sourceQueue) {};
	};

	template <typename T, typename... Args>
	class ConcreteResource: Resource<T> {
		std::tuple<Args...> stored_args;

	public:
		explicit ConcreteResource(Args&&... args): stored_args(std::forward<Args>(args)...) {}

		T initialize() {
			return std::apply(
				[](auto&&... args) { return T(std::forward<decltype(args)>(args)...); },
				std::move(stored_args)
			);
		}
	}; // Recipie should have a list of concrete resource requirements

	class Recipe {};

	class PreviousFrame: public Node {};

	class NextFrame: public Node {};

	class FrameStart: public Phase {};

	class FrameEnd: public Phase {};

	class Operation {};

	template <typename T>
	class Create: public operation {};

	template <typename T>
	class Read: public operation {};

	template <typename T, typename X = T>
	class Modify: public operation {};

	template <typename T, typename X = T>
	class Transform: public operation {};
}; // namespace Brassica

#include <vk_mem_alloc.h>

#include <memory>
#include <stdexcept>

#include <vulkan/vulkan_hpp.hpp>

struct TextureOptions {
	vk::Format          format = vk::Format::eR8G8B8A8Srgb;
	vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageTiling     tiling = vk::ImageTiling::eOptimal;
	uint32_t            mipLevels = 1;

	// Memory settings
	VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationCreateFlags memoryFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

	// View specific overrides
	vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor;
	vk::ComponentMapping components = vk::ComponentMapping{}; // Identity mapping by default
};

class Texture2D {
public:
	Texture2D(vk::Device device, VmaAllocator allocator, uint32_t width, uint32_t height, TextureOptions options = {}):
		m_device(device), m_allocator(allocator), m_width(width), m_height(height), m_format(options.format) {
		// 1. Build and create the Vulkan Image with VMA memory allocation
		vk::ImageCreateInfo imageInfo(
			{},
			vk::ImageType::e2D,
			m_format,
			vk::Extent3D(width, height, 1),
			options.mipLevels,
			1, // Array layers
			vk::SampleCountFlagBits::e1,
			options.tiling,
			options.usage,
			vk::SharingMode::eExclusive
		);

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = options.memoryUsage;
		allocInfo.flags = options.memoryFlags;

		VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
		VkImage           rawImage;
		if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate texture image via VMA!");
		}
		m_image = rawImage;

		// 2. Automatically generate the companion Image View
		vk::ImageViewCreateInfo viewInfo(
			{},
			m_image,
			vk::ImageViewType::e2D,
			m_format,
			options.components,
			vk::ImageSubresourceRange(options.aspectFlags, 0, options.mipLevels, 0, 1)
		);

		// Creates the View handle directly using Vulkan-Hpp bindings
		m_view = m_device.createImageView(viewInfo);
	}

	// Clean up handles in reverse order of creation
	~Texture2D() {
		if (m_view) {
			m_device.destroyImageView(m_view);
		}
		if (m_image && m_allocation) {
			vmaDestroyImage(m_allocator, m_image, m_allocation);
		}
	}

	// Explicitly delete copy semantics
	Texture2D(const Texture2D&) = delete;
	Texture2D& operator=(const Texture2D&) = delete;

	// Support move semantics cleanly
	Texture2D(Texture2D&& other) noexcept:
		m_device(other.m_device),
		m_allocator(other.m_allocator),
		m_image(other.m_image),
		m_allocation(other.m_allocation),
		m_view(other.m_view),
		m_width(other.m_width),
		m_height(other.m_height),
		m_format(other.m_format) {
		other.m_image = nullptr;
		other.m_allocation = nullptr;
		other.m_view = nullptr;
	}

	// Setters / Getters
	vk::Image GetImage() const { return m_image; }

	vk::ImageView GetImageView() const { return m_view; }

	VmaAllocation GetAllocation() const { return m_allocation; }

	vk::Format GetFormat() const { return m_format; }

private:
	vk::Device   m_device = nullptr;
	VmaAllocator m_allocator = nullptr;

	vk::Image     m_image = nullptr;
	VmaAllocation m_allocation = nullptr;
	vk::ImageView m_view = nullptr;

	uint32_t   m_width;
	uint32_t   m_height;
	vk::Format m_format;
};

//////////////

#include <vk_mem_alloc.h>

#include <memory>
#include <stdexcept>

#include <vulkan/vulkan_hpp.hpp>

struct TextureOptions {
	vk::Format          format = vk::Format::eR8G8B8A8Srgb;
	vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::ImageTiling     tiling = vk::ImageTiling::eOptimal;
	uint32_t            mipLevels = 1;

	VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
	VmaAllocationCreateFlags memoryFlags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

	vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor;
	vk::ComponentMapping components = vk::ComponentMapping{};
};

class Texture2D {
public:
	Texture2D(vk::Device device, VmaAllocator allocator, uint32_t width, uint32_t height, TextureOptions options = {}):
		m_device(device), m_allocator(allocator), m_width(width), m_height(height), m_format(options.format) {
		// 1. Build and create the Vulkan Image with VMA memory allocation
		vk::ImageCreateInfo imageInfo(
			{},
			vk::ImageType::e2D,
			m_format,
			vk::Extent3D(width, height, 1),
			options.mipLevels,
			1,
			vk::SampleCountFlagBits::e1,
			options.tiling,
			options.usage,
			vk::SharingMode::eExclusive
		);

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = options.memoryUsage;
		allocInfo.flags = options.memoryFlags;

		VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
		VkImage           rawImage;
		if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate texture image via VMA!");
		}
		m_image = rawImage;

		// 2. Automatically generate the companion Image View
		vk::ImageViewCreateInfo viewInfo(
			{},
			m_image,
			vk::ImageViewType::e2D,
			m_format,
			options.components,
			vk::ImageSubresourceRange(options.aspectFlags, 0, options.mipLevels, 0, 1)
		);

		m_view = m_device.createImageView(viewInfo);
	}

	~Texture2D() {
		if (m_view)
			m_device.destroyImageView(m_view);
		if (m_image && m_allocation)
			vmaDestroyImage(m_allocator, m_image, m_allocation);
		if (m_sampler)
			CleanupSampler();
	}

	// Move-only operations
	Texture2D(const Texture2D&) = delete;
	Texture2D& operator=(const Texture2D&) = delete;

	Texture2D(Texture2D&& other) noexcept:
		m_device(other.m_device),
		m_allocator(other.m_allocator),
		m_image(other.m_image),
		m_allocation(other.m_allocation),
		m_view(other.m_view),
		m_width(other.m_width),
		m_height(other.m_height),
		m_format(other.m_format) {
		other.m_image = nullptr;
		other.m_allocation = nullptr;
		other.m_view = nullptr;
	}

	// Pipeline Data Upload Function
	void UploadPixelData(vk::CommandBuffer cmdBuffer, VmaAllocator allocator, const void* pixelData, size_t dataSize) {
		// A. Allocate a temporary, host-visible staging buffer via VMA
		vk::BufferCreateInfo
			bufferInfo({}, dataSize, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive);
		VmaAllocationCreateInfo stagingAllocInfo{};
		stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
			VMA_ALLOCATION_CREATE_MAPPED_BIT;

		VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
		VkBuffer           rawStagingBuffer;
		VmaAllocation      stagingAllocation;
		VmaAllocationInfo  stagingAllocData;

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

		// B. Memory copy your raw pixel array into the mapped CPU pointer
		std::memcpy(stagingAllocData.pMappedData, pixelData, dataSize);

		// C. Record layout barrier: Undefined -> Transfer Dst Optimal
		vk::ImageMemoryBarrier barrierToTransfer(
			vk::AccessFlagBits::eNone,
			vk::AccessFlagBits::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			m_image,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		);

		cmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe,
			vk::PipelineStageFlagBits::eTransfer,
			{},
			nullptr,
			nullptr,
			barrierToTransfer
		);

		// D. Record the Buffer-to-Image copy transaction
		vk::BufferImageCopy copyRegion(
			0,
			0,
			0,
			vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
			vk::Offset3D(0, 0, 0),
			vk::Extent3D(m_width, m_height, 1)
		);

		cmdBuffer.copyBufferToImage(rawStagingBuffer, m_image, vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

		// E. Record layout barrier: Transfer Dst Optimal -> Shader Read Only Optimal
		vk::ImageMemoryBarrier barrierToShader(
			vk::AccessFlagBits::eTransferWrite,
			vk::AccessFlagBits::eShaderRead,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			VK_QUEUE_FAMILY_IGNORED,
			VK_QUEUE_FAMILY_IGNORED,
			m_image,
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		);

		cmdBuffer.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer,
			vk::PipelineStageFlagBits::eFragmentShader,
			{},
			nullptr,
			nullptr,
			barrierToShader
		);

		// F. Schedule clean up of the staging buffer after the GPU pipeline processing completes
		// Note: In an engine loop, you must defer 'vmaDestroyBuffer' until after this frame's fence is signaled!
		m_deferredStagingBuffers.push_back({rawStagingBuffer, stagingAllocation});
	}

	// Generates a boilerplate texture sampler with sensible filtering defaults
	void CreateDefaultSampler(vk::Filter filter = vk::Filter::eLinear) {
		vk::SamplerCreateInfo samplerInfo(
			{},
			filter,
			filter, // Mag and Min filters
			vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			vk::SamplerAddressMode::eRepeat,
			0.0f,
			VK_FALSE,
			1.0f, // Anisotropy disabled by default
			VK_FALSE,
			vk::CompareOp::eAlways,
			0.0f,
			0.0f,
			vk::BorderColor::eIntOpaqueBlack,
			VK_FALSE
		);
		m_sampler = m_device.createSampler(samplerInfo);
	}

	// Explicitly write our image resources into a target Descriptor Set
	void BindToDescriptorSet(vk::DescriptorSet targetSet, uint32_t bindingIndex = 0) {
		if (!m_sampler) {
			CreateDefaultSampler(); // Guarantee we have a sampler layout active
		}

		// Struct containing the concrete texture asset bindings
		vk::DescriptorImageInfo imageInfo(
			m_sampler,
			m_view,
			vk::ImageLayout::eShaderReadOnlyOptimal // The layout we transitioned to during upload
		);

		// Record the layout update instruction
		vk::WriteDescriptorSet writeInfo(
			targetSet,
			bindingIndex,
			0, // Array element offset
			1, // Descriptor count
			vk::DescriptorType::eCombinedImageSampler,
			&imageInfo,
			nullptr,
			nullptr
		);

		m_device.updateDescriptorSets(1, &writeInfo, 0, nullptr);
	}

	// Clean up inside destructor
	void CleanupSampler() {
		if (m_sampler) {
			m_device.destroySampler(m_sampler);
			m_sampler = nullptr;
		}
	}

	vk::Sampler GetSampler() const { return m_sampler; }

	void CleanupStagingResources() {
		for (auto& res : m_deferredStagingBuffers) {
			vmaDestroyBuffer(m_allocator, res.first, res.second);
		}
		m_deferredStagingBuffers.clear();
	}

	// Getters
	vk::Image GetImage() const { return m_image; }

	vk::ImageView GetImageView() const { return m_view; }

private:
	vk::Device    m_device;
	VmaAllocator  m_allocator;
	vk::Image     m_image;
	VmaAllocation m_allocation;
	vk::ImageView m_view;
	uint32_t      m_width;
	uint32_t      m_height;
	vk::Format    m_format;
	vk::Sampler   m_sampler = nullptr;

	std::vector<std::pair<VkBuffer, VmaAllocation>> m_deferredStagingBuffers;
};

#include <stdexcept>
#include <vector>

#include <vulkan/vulkan_hpp.hpp>

struct DescriptorLayoutOptions {
	uint32_t             bindingIndex = 0;
	vk::DescriptorType   type = vk::DescriptorType::eCombinedImageSampler;
	uint32_t             count = 1;
	vk::ShaderStageFlags stageFlags = vk::ShaderStageFlagBits::eFragment;
};

class DescriptorSetManager {
public:
	DescriptorSetManager(vk::Device device, DescriptorLayoutOptions options = {}): m_device(device) {
		// 1. Define the structural binding configuration
		vk::DescriptorSetLayoutBinding
			layoutBinding(options.bindingIndex, options.type, options.count, options.stageFlags, nullptr);

		vk::DescriptorSetLayoutCreateInfo layoutInfo({}, 1, &layoutBinding);
		m_layout = m_device.createDescriptorSetLayout(layoutInfo);

		// 2. Build a descriptor pool sized specifically for allocating one set
		vk::DescriptorPoolSize       poolSize(options.type, options.count);
		vk::DescriptorPoolCreateInfo poolInfo(
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, // Allows resetting individual sets
			1,                                                    // Max sets from this pool
			1,
			&poolSize
		);
		m_pool = m_device.createDescriptorPool(poolInfo);
	}

	~DescriptorSetManager() {
		if (m_pool)
			m_device.destroyDescriptorPool(m_pool);
		if (m_layout)
			m_device.destroyDescriptorSetLayout(m_layout);
	}

	// Allocate an empty Descriptor Set using our layout
	vk::DescriptorSet AllocateSet() {
		vk::DescriptorSetAllocateInfo allocInfo(m_pool, 1, &m_layout);

		vk::DescriptorSet descriptorSet;
		if (m_device.allocateDescriptorSets(&allocInfo, &descriptorSet) != vk::Result::eSuccess) {
			throw std::runtime_error("Failed to allocate descriptor set!");
		}
		return descriptorSet;
	}

	// Getters
	vk::DescriptorSetLayout GetLayout() const { return m_layout; }

private:
	vk::Device              m_device;
	vk::DescriptorSetLayout m_layout;
	vk::DescriptorPool      m_pool;
};

/////////////////////

///////////////////

#include <concepts>
#include <iostream>
#include <string_view>

// ==========================================
// 1. The Tree-Wide Sequence Counter
// ==========================================
namespace counter_impl {
	template <size_t N>
	struct tag: tag<N - 1> {};

	template <>
	struct tag<0> {};

	template <size_t N, typename RootContext>
	struct flag {
		friend constexpr bool is_defined(flag<N, RootContext>, tag<N>);
	};

	template <size_t N, typename RootContext>
	struct writer {
		friend constexpr bool is_defined(flag<N, RootContext>, tag<N>) { return true; }
	};

	template <size_t N, typename RootContext>
	constexpr size_t next_id(...) {
		return N;
	}

	template <size_t N, typename RootContext>
		requires requires { is_defined(flag<N, RootContext>{}, tag<N>{}); }
	constexpr size_t next_id(int) {
		return next_id<N + 1, RootContext>(0);
	}
} // namespace counter_impl

template <typename RootContext, size_t UniqueID = counter_impl::next_id<0, RootContext>(0)>
constexpr int assign_tree_id() {
	[[maybe_unused]] auto w = counter_impl::writer<UniqueID, RootContext>{};
	return static_cast<int>(UniqueID);
}

// ==========================================
// 2. The TypeSafeEnum Structure
// ==========================================
// We template against a "Root" type so that parents and children share the exact same C++ type.
template <typename Root>
class TypeSafeEnum {
private:
	int value;

public:
	// Every time an enum instance is instantiated, it grabs the next ID from the Root counter.
	constexpr explicit TypeSafeEnum(int val = assign_tree_id<Root>()): value(val) {}

	constexpr int underlying() const { return value; }

	auto operator<=>(const TypeSafeEnum&) const = default;
};

// ==========================================
// 3. Defining the Enum Tree
// ==========================================
// Base struct acting as the Type Anchor
struct NetworkErrorRoot {};

using NetworkError = TypeSafeEnum<NetworkErrorRoot>;

struct NetworkErrorDef {
	inline static constexpr std::string_view typeName = "NetworkError";

	// Values share the NetworkErrorRoot type context
	inline static constexpr NetworkError Timeout;
	inline static constexpr NetworkError Disconnected;
};

// Extension inherits from base definitions but adds new entries
struct ExtendedNetworkErrorDef: public NetworkErrorDef {
	inline static constexpr std::string_view typeName = "ExtendedNetworkError";

	// These STILL use 'NetworkError' type! They seamlessly interoperate.
	inline static constexpr NetworkError RateLimited;
	inline static constexpr NetworkError ServerError;
};

/*
#include <iostream>
#include <concepts>
#include <string_view>

// ==========================================
// 1. The Stateful Metaprogramming Counter
// ==========================================
namespace counter_impl {
    template <size_t N> struct tag : tag<N - 1> {};
    template <> struct tag<0> {};

    template <size_t N, typename Context>
    struct flag {
        friend constexpr bool is_defined(flag<N, Context>, tag<N>);
    };

    template <size_t N, typename Context>
    struct writer {
        friend constexpr bool is_defined(flag<N, Context>, tag<N>) { return true; }
    };

    template <size_t N, typename Context>
    constexpr size_t next_id(...) {
        return N;
    }

    template <size_t N, typename Context>
    requires requires { is_defined(flag<N, Context>{}, tag<N>{}); }
    constexpr size_t next_id(int) {
        return next_id<N + 1, Context>(0);
    }
}

// Helper macro/function to fetch and increment the compile-time counter for a Type context
template <typename Context, size_t UniqueID = counter_impl::next_id<0, Context>(0)>
constexpr int next_value() {
    [[maybe_unused]] auto w = counter_impl::writer<UniqueID, Context>{};
    return static_cast<int>(UniqueID);
}

// ==========================================
// 2. The Refactored TypeSafeEnum Framework
// ==========================================
template <typename T>
concept IsEnumDefinition = requires {
    { T::typeName } -> std::convertible_to<std::string_view>;
};

template <IsEnumDefinition Def>
class TypeSafeEnum {
private:
    int value;

public:
    // C++ requires default arguments to be evaluated at the call-site.
    // By passing next_value<Def>() as a default template-driven argument,
    // each distinct instantiation of this constructor locks in a brand new sequential ID.
    constexpr explicit TypeSafeEnum(int val = next_value<Def>()) : value(val) {}

    friend Def;
    constexpr int underlying() const { return value; }
    auto operator<=>(const TypeSafeEnum&) const = default;
};

// ==========================================
// 3. Exact Usage As You Intended
// ==========================================
struct NetworkErrorDef {
    inline static constexpr std::string_view typeName = "NetworkError";

    using Enum = TypeSafeEnum<NetworkErrorDef>;

    // Evaluated sequentially: next_value<NetworkErrorDef>() returns 0, then 1
    inline static constexpr Enum Timeout;
    inline static constexpr Enum Disconnected;
};
using NetworkError = NetworkErrorDef::Enum;

struct ExtendedNetworkErrorDef : public NetworkErrorDef {
    inline static constexpr std::string_view typeName = "ExtendedNetworkError";

    using Enum = TypeSafeEnum<ExtendedNetworkErrorDef>;

    // Evaluated sequentially: next_value<ExtendedNetworkErrorDef>() starts fresh at 0, then 1
    inline static constexpr Enum RateLimited;
    inline static constexpr Enum ServerError;
};
using ExtendedNetworkError = ExtendedNetworkErrorDef::Enum;

// // ==========================================
// // 4. Verification
// // ==========================================
// int main() {
//     std::cout << NetworkErrorDef::typeName << " values:\n";
//     std::cout << "  Timeout:      " << NetworkError::Timeout.underlying() << "\n";
//     std::cout << "  Disconnected: " << NetworkError::Disconnected.underlying() << "\n\n";

//     std::cout << ExtendedNetworkErrorDef::typeName << " values:\n";
//     std::cout << "  RateLimited:  " << ExtendedNetworkError::RateLimited.underlying() << "\n";
//     std::cout << "  ServerError:  " << ExtendedNetworkError::ServerError.underlying() << "\n";

//     // True compile-time verification
//     static_assert(NetworkError::Timeout.underlying() == 0);
//     static_assert(NetworkError::Disconnected.underlying() == 1);
//     static_assert(ExtendedNetworkError::RateLimited.underlying() == 0);
//     static_assert(ExtendedNetworkError::ServerError.underlying() == 1);
// }


// #include <iostream>
// #include <concepts>
// #include <string_view>

// template <typename T>
// concept IsEnumDefinition = requires {
//     { T::typeName } -> std::convertible_to<std::string_view>;
// };

// template <IsEnumDefinition Def>
// class TypeSafeEnum {
// private:
// 	inline static constexpr int counter = 0;
//     int value;

//     constexpr explicit TypeSafeEnum(int val = counter++) : value(val) {}

// public:
//     friend Def;
//     constexpr int underlying() const { return value; }
//     auto operator<=>(const TypeSafeEnum&) const = default;
// };

// struct NetworkErrorDef {
//     inline static constexpr std::string_view typeName = "NetworkError";

//     using Enum = TypeSafeEnum<NetworkErrorDef>;

//     inline static constexpr Enum Timeout;
//     inline static constexpr Enum Disconnected;
// };
// using NetworkError = NetworkErrorDef::Enum;

// struct ExtendedNetworkErrorDef : public NetworkErrorDef {
//     inline static constexpr std::string_view typeName = "ExtendedNetworkError";

//     using Enum = TypeSafeEnum<ExtendedNetworkErrorDef>;

//     inline static constexpr Enum RateLimited;
//     inline static constexpr Enum ServerError;
// };
// using ExtendedNetworkError = ExtendedNetworkErrorDef::Enum;


*/

// // ==========================================
// // 2. The TypeSafeEnum Structure
// // ==========================================
// // We template against a "Root" type so that parents and children share the exact same C++ type.
// template <typename Root>
// class TypeSafeEnum {
// private:
//     int value;
//     std::string_view name;

// public:
//     // Every time an enum instance is instantiated, it grabs the next ID from the Root counter.
//     constexpr explicit TypeSafeEnum(std::string_view n, int val = assign_tree_id<Root>())
//         : value(val), name(n) {}

//     constexpr int underlying() const { return value; }
//     constexpr std::string_view to_string() const { return name; }

//     auto operator<=>(const TypeSafeEnum&) const = default;
// };

// // ==========================================
// // 3. Defining the Enum Tree
// // ==========================================
// // Base struct acting as the Type Anchor
// struct NetworkErrorRoot {};
// using NetworkError = TypeSafeEnum<NetworkErrorRoot>;

// struct NetworkErrorDef {
//     inline static constexpr std::string_view typeName = "NetworkError";

//     // Values share the NetworkErrorRoot type context
//     inline static constexpr NetworkError Timeout{"Timeout"};
//     inline static constexpr NetworkError Disconnected{"Disconnected"};
// };

// // Extension inherits from base definitions but adds new entries
// struct ExtendedNetworkErrorDef : public NetworkErrorDef {
//     inline static constexpr std::string_view typeName = "ExtendedNetworkError";

//     // These STILL use 'NetworkError' type! They seamlessly interoperate.
//     inline static constexpr NetworkError RateLimited{"RateLimited"};
//     inline static constexpr NetworkError ServerError{"ServerError"};
// };

/*


    // A template that holds a type T and the arguments needed to construct it
    template <typename T, typename... Args>
    class LazyInitializer {
    private:
        // Store the arguments as a tuple
        std::tuple<Args...> stored_args;

    public:
        // Constructor uses perfect forwarding to capture arguments efficiently
        explicit LazyInitializer(Args&&... args): stored_args(std::forward<Args>(args)...) {}

        // Method to create and return the object T using the stored arguments
        T initialize() {
            // std::apply unpacks the tuple into the constructor of T
            return std::apply(
                [](auto&&... args) { return T(std::forward<decltype(args)>(args)...); },
                std::move(stored_args)
            );
        }
    };


class Graph {
    bool m_hasActiveInternalNodes = true;

public:
    // Fulfills NodeLike Concept: Setup Phase
    Recipe Setup(const RenderContext& ctx) {
        // 1. Iterate internal nodes and call Setup()
        // 2. Perform internal topological sort and culling
        // 3. Determine if the subgraph as a whole has work to do

        return Recipe{
            .domain = ExecutionDomain::Graphics, // See domain mapping note below
            .isActive = m_hasActiveInternalNodes
        };
    }

    // Fulfills NodeLike Concept: Execution Phase
    void Execute(CommandBuffer& cmd) {
        if (!m_hasActiveInternalNodes)
            return;

        // Iterate surviving internal nodes and pass the parent's
        // command buffer down the chain.
        // for (auto& node : m_sortedActiveNodes) { node.Execute(cmd); }
    }

    template <NodeLike T, typename... Ops>
    void RegisterNode() {}
};

// A compile-time assertion to guarantee Graph always satisfies NodeLike
static_assert(NodeLike<Graph>, "Graph must fulfill NodeLike to act as a subgraph");

// 1. Define the possible inputs using a std::variant
using InputVar = std::variant<int, double, std::string>;

// 2. The Worker Class (advertises what it wants, receives them in an array/vector)
class MyWorker {
public:
    // This defines the "type tags" the class expects to receive
    using ExpectedTypes = std::tuple<int, std::string>;

    // The method that accepts the variable set of inputs matching the tags
    void execute(int id, const std::string& name) {
        std::cout << "Worker executed with ID: " << id << " and Name: " << name << "\n";
    }
};

// 3. The Coordinator Class (extracts the right types from a pool of inputs)
class Coordinator {
public:
    template <typename Worker>
    static void dispatch(Worker& worker, const std::vector<InputVar>& input_pool) {
        // Extract the tuple of type tags from the worker
        typename Worker::ExpectedTypes tags;

        // Unpack the tuple tags at compile-time and extract them from the input pool
        std::apply(
            [&worker, &input_pool](auto... dummy_tags) {
                // Lambda to extract a specific type 'T' from the generic input pool
                auto extract = [&input_pool](auto tag_type) {
                    using T = decltype(tag_type);
                    for (const auto& var : input_pool) {
                        if (std::holds_alternative<T>(var)) {
                            return std::get<T>(var);
                        }
                    }
                    throw std::runtime_error("Required type not found in input pool!");
                };

                // Call the execute method by passing the extracted arguments
                worker.execute(extract(dummy_tags)...);
            },
            tags
        );
    }
};

// frameGraph.hpp
#pragma once
#include <concepts>
#include <type_traits>

namespace Brassica {

    // 1. Strong Typing for Keys
    // Users define empty structs like `struct GBufferAlbedo {};` to act as keys.
    // This ensures the compiler catches resource mismatches.
    template <typename T>
    concept ResourceKey = std::is_empty_v<T> && std::is_trivially_constructible_v<T>;

    // 2. Execution Domains
    // Replaces the empty Queue class with a strongly typed enum for Vulkan queue mapping.
    enum class ExecutionDomain { Graphics, Compute, Host, Transfer };

    // 3. Synchronization & Execution Stubs
    // Analogues for Vulkan 1.3 VkImageMemoryBarrier2 and VkCommandBuffer
    struct MemoryBarrier {};

    struct CommandBuffer {};

    struct RenderContext {}; // Carries frame state (e.g., UI toggles, camera data)

    // 4. Recipe Definition
    // Returned dynamically by node factories to indicate current frame requirements.
    struct Recipe {
        ExecutionDomain domain = ExecutionDomain::Graphics;
        bool            isActive = true;
    };

    // 5. Node Factory Concept
    // Enforces that anything treated as a node can evaluate context to return a recipe,
    // and can execute commands when scheduled.
    template <typename T>
    concept NodeLike = requires(T t, const RenderContext& ctx, CommandBuffer& cmd) {
        { t.Setup(ctx) } -> std::same_as<Recipe>;
        { t.Execute(cmd) };
    };

    // 6. Phases & Ordering
    class Phase {};

    class FrameStart: public Phase {};

    class FrameEnd: public Phase {};

    class PreviousFrame {};

    class NextFrame {};

    // 7. Graph Container
    class Graph {
    public:
        // Registers a node factory, inferring input/output dependencies at compile time via Operations
        template <NodeLike T, typename... Ops>
        void RegisterNode() {}

        // Injects external resources (like the swapchain) directly via type keys
        template <ResourceKey K>
        void InjectExternalResource(class Resource* resource) {}

        // The three-phase frame loop
        void Setup(const RenderContext& ctx) {}

        void Compile() {}

        void Execute() {}
    };

    // 8. Operations (Constrained to Resource Keys)
    class Operation {};

    template <ResourceKey T>
    class Create: public Operation {
        using Key = T;
    };

    template <ResourceKey T>
    class Read: public Operation {
        using Key = T;
    };

    template <ResourceKey T, ResourceKey X = T>
    class Modify: public Operation {
        using ReadKey = T;
        using WriteKey = X;
    };

    // 9. Resource Interface
    // Returns barrier metadata structs for Graph-level batching instead of executing commands directly.
    class Resource {
    public:
        virtual ~Resource() = default;
        virtual MemoryBarrier GetReadBarrier() const = 0;
        virtual MemoryBarrier GetWriteBarrier() const = 0;
        virtual MemoryBarrier GetReleaseBarrier(ExecutionDomain destQueue) const = 0;
        virtual MemoryBarrier GetAcquireBarrier(ExecutionDomain sourceQueue) const = 0;
    };

}; // namespace Brassica



#include <iostream>
#include <tuple>
#include <variant>
#include <vector>
#include <string>

// 1. Define the possible inputs using a std::variant
using InputVar = std::variant<int, double, std::string>;

// 2. The Worker Class (advertises what it wants, receives them in an array/vector)
class MyWorker {
public:
    // This defines the "type tags" the class expects to receive
    using ExpectedTypes = std::tuple<int, std::string>;

    // The method that accepts the variable set of inputs matching the tags
    void execute(int id, const std::string& name) {
        std::cout << "Worker executed with ID: " << id << " and Name: " << name << "\n";
    }
};

// 3. The Coordinator Class (extracts the right types from a pool of inputs)
class Coordinator {
public:
    template <typename Worker>
    static void dispatch(Worker& worker, const std::vector<InputVar>& input_pool) {
        // Extract the tuple of type tags from the worker
        typename Worker::ExpectedTypes tags;

        // Unpack the tuple tags at compile-time and extract them from the input pool
        std::apply([&worker, &input_pool](auto... dummy_tags) {

            // Lambda to extract a specific type 'T' from the generic input pool
            auto extract = [&input_pool](auto tag_type) {
                using T = decltype(tag_type);
                for (const auto& var : input_pool) {
                    if (std::holds_alternative<T>(var)) {
                        return std::get<T>(var);
                    }
                }
                throw std::runtime_error("Required type not found in input pool!");
            };

            // Call the execute method by passing the extracted arguments
            worker.execute(extract(dummy_tags)...);

        }, tags);
    }
};


*/