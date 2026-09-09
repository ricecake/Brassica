#include <vk_mem_alloc.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vulkan/vulkan.hpp"

/* USAGE

// Initialize context container
AssetManager asset(device, allocator);

// 1. Allocate a standard volumetric lookup table (3D Texture)
asset.Create3DTexture("volumetric_fog_lut", 256, 256, 64);

// 2. Setup standard render attachments with automatic format handling
asset.CreateDepthBuffer("main_frame_depth", 1920, 1080);
asset.CreateComputeShaderTarget("raytraced_shadows_out", 1920, 1080);

// 3. Spin up your structural vertex mesh memory array
struct Vertex { float pos[3]; float uv[2]; };
asset.CreateVertexBuffer("quad_vertices", sizeof(Vertex) * 4);

// 4. Easily update frame transforms on every update loop frame pass
struct CameraMatrices { glm::mat4 viewProj; } uboData;
BufferResource* myUbo = asset.GetBuffer("main_camera_ubo");
myUbo->UpdateData(&uboData, sizeof(CameraMatrices));


*/

// Semantic aliases for common image roles
enum class ImageUsagePurpose {
	StandardTexture,     // Read by fragment shaders, populated by CPU
	DepthBuffer,         // Depth/Stencil attachment (GPU only)
	ColorAttachment,     // Render target for offscreen rendering
	ComputeShaderTarget, // Storage image read/written by a Compute shader
	DataReadback         // GPU-written data to be read back by CPU (screenshots, etc.)
};

// Semantic aliases for buffer roles
enum class BufferUsagePurpose {
	VertexInput,    // High-speed GPU memory for raw vertex data
	IndexInput,     // High-speed GPU memory for index arrays
	UniformBlock,   // Constant data frequently updated/read by shaders
	StorageCompute, // Structured buffers for compute or heavy data storage
	StagingTransfer // CPU-visible staging memory
};

// Flexible base options for Images (supporting 2D and 3D dimensions)
struct ImageConfigOptions {
	vk::Format      format = vk::Format::eR8G8B8A8Srgb;
	vk::ImageTiling tiling = vk::ImageTiling::eOptimal;
	uint32_t        mipLevels = 1;
	uint32_t        depth = 1; // Used for 3D textures, defaults to 1 for 2D

	// Explicit override flags (leave as eNone if using a purpose-built alias)
	vk::ImageUsageFlags      customUsageFlags = vk::ImageUsageFlags(0);
	VmaMemoryUsage           customMemoryUsage = VMA_MEMORY_USAGE_UNKNOWN;
	VmaAllocationCreateFlags customAllocFlags = VmaAllocationCreateFlags(0);
};

class TextureResource {
public:
	TextureResource(
		vk::Device         device,
		VmaAllocator       allocator,
		uint32_t           width,
		uint32_t           height,
		ImageUsagePurpose  purpose,
		ImageConfigOptions options = {}
	):
		m_device(device),
		m_allocator(allocator),
		m_width(width),
		m_height(height),
		m_depth(options.depth),
		m_format(options.format) {
		// 1. Resolve Vulkan & VMA configurations based on structural intent/purpose
		vk::ImageUsageFlags      usageFlags;
		VmaMemoryUsage           memoryUsage = VMA_MEMORY_USAGE_AUTO;
		VmaAllocationCreateFlags allocFlags = 0;
		vk::ImageAspectFlags     aspectFlags = vk::ImageAspectFlagBits::eColor;

		switch (purpose) {
		case ImageUsagePurpose::StandardTexture:
			usageFlags = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
			break;
		case ImageUsagePurpose::DepthBuffer:
			usageFlags = vk::ImageUsageFlagBits::eDepthStencilAttachment;
			aspectFlags = vk::ImageAspectFlagBits::eDepth;
			// Auto-fallback to depth format if the user didn't specify one
			if (m_format == vk::Format::eR8G8B8A8Srgb)
				m_format = vk::Format::eD32Sfloat;
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

		// Apply overrides if provided
		if (options.customUsageFlags)
			usageFlags = options.customUsageFlags;
		if (options.customMemoryUsage != VMA_MEMORY_USAGE_UNKNOWN)
			memoryUsage = options.customMemoryUsage;
		if (options.customAllocFlags)
			allocFlags = options.customAllocFlags;

		// 2. Identify Dimensionality (2D vs 3D)
		vk::ImageType     imageType = (m_depth > 1) ? vk::ImageType::e3D : vk::ImageType::e2D;
		vk::ImageViewType viewType = (m_depth > 1) ? vk::ImageViewType::e3D : vk::ImageViewType::e2D;

		// 3. Allocate via VMA
		vk::ImageCreateInfo imageInfo(
			{},
			imageType,
			m_format,
			vk::Extent3D(width, height, m_depth),
			options.mipLevels,
			1,
			vk::SampleCountFlagBits::e1,
			options.tiling,
			usageFlags,
			vk::SharingMode::eExclusive
		);

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = memoryUsage;
		allocInfo.flags = allocFlags;

		VkImageCreateInfo cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
		VkImage           rawImage;
		if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate image resource!");
		}
		m_image = rawImage;

		// 4. Instantiate matching View layout automatically
		vk::ImageViewCreateInfo viewInfo(
			{},
			m_image,
			viewType,
			m_format,
			{},
			vk::ImageSubresourceRange(aspectFlags, 0, options.mipLevels, 0, 1)
		);
		m_view = m_device.createImageView(viewInfo);
	}

	~TextureResource() {
		if (m_view)
			m_device.destroyImageView(m_view);
		if (m_image && m_allocation)
			vmaDestroyImage(m_allocator, m_image, m_allocation);
	}

	// Move semantic lifecycle hooks
	TextureResource(const TextureResource&) = delete;
	TextureResource& operator=(const TextureResource&) = delete;

	TextureResource(TextureResource&& o) noexcept:
		m_device(o.m_device),
		m_allocator(o.m_allocator),
		m_image(o.m_image),
		m_allocation(o.m_allocation),
		m_view(o.m_view),
		m_width(o.m_width),
		m_height(o.m_height),
		m_depth(o.m_depth),
		m_format(o.m_format) {
		o.m_image = nullptr;
		o.m_allocation = nullptr;
		o.m_view = nullptr;
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
	uint32_t      m_width, m_height, m_depth;
	vk::Format    m_format;
};

class BufferResource {
public:
	BufferResource(vk::Device device, VmaAllocator allocator, size_t size, BufferUsagePurpose purpose):
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
			// Persistent mapping lets CPU write data continuously without flushing maps
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

		vk::BufferCreateInfo    bufferInfo({}, size, usageFlags, vk::SharingMode::eExclusive);
		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.flags = allocFlags;

		VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
		VkBuffer           rawBuffer;
		if (vmaCreateBuffer(m_allocator, &cBufferInfo, &allocInfo, &rawBuffer, &m_allocation, &m_allocInfo) !=
		    VK_SUCCESS) {
			throw std::runtime_error("Failed to allocate uniform/vertex buffer!");
		}
		m_buffer = rawBuffer;
	}

	~BufferResource() {
		if (m_buffer && m_allocation)
			vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
	}

	// Move assignment/constructor overrides
	BufferResource(const BufferResource&) = delete;
	BufferResource& operator=(const BufferResource&) = delete;

	BufferResource(BufferResource&& o) noexcept:
		m_device(o.m_device),
		m_allocator(o.m_allocator),
		m_buffer(o.m_buffer),
		m_allocation(o.m_allocation),
		m_allocInfo(o.m_allocInfo),
		m_size(o.m_size) {
		o.m_buffer = nullptr;
		o.m_allocation = nullptr;
	}

	// Direct access helper for persistently mapped memory blocks (Uniforms)
	void UpdateData(const void* srcData, size_t size, size_t offset = 0) {
		if (!m_allocInfo.pMappedData)
			throw std::runtime_error("Cannot direct-write to an unmapped buffer!");
		std::memcpy(static_cast<char*>(m_allocInfo.pMappedData) + offset, srcData, size);
	}

	vk::Buffer GetBuffer() const { return m_buffer; }

private:
	vk::Device        m_device;
	VmaAllocator      m_allocator;
	vk::Buffer        m_buffer = nullptr;
	VmaAllocation     m_allocation = nullptr;
	VmaAllocationInfo m_allocInfo{};
	size_t            m_size;
};

class AssetManager {
public:
	AssetManager(vk::Device device, VmaAllocator allocator): m_device(device), m_allocator(allocator) {}

	// --- TEXTURE ARCHITECTURAL FACTORIES ---

	TextureResource*
	Create2DTexture(const std::string& name, uint32_t width, uint32_t height, ImageConfigOptions opts = {}) {
		auto [it, success] = m_textures.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, width, height, ImageUsagePurpose::StandardTexture, opts)
		);
		return &it->second;
	}

	TextureResource* Create3DTexture(
		const std::string& name,
		uint32_t           width,
		uint32_t           height,
		uint32_t           depth,
		ImageConfigOptions opts = {}
	) {
		opts.depth = depth; // Enforce depth sizing dimension
		auto [it, success] = m_textures.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, width, height, ImageUsagePurpose::StandardTexture, opts)
		);
		return &it->second;
	}

	TextureResource*
	CreateDepthBuffer(const std::string& name, uint32_t width, uint32_t height, ImageConfigOptions opts = {}) {
		auto [it, success] = m_textures.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, width, height, ImageUsagePurpose::DepthBuffer, opts)
		);
		return &it->second;
	}

	TextureResource*
	CreateComputeShaderTarget(const std::string& name, uint32_t width, uint32_t height, ImageConfigOptions opts = {}) {
		auto [it, success] = m_textures.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, width, height, ImageUsagePurpose::ComputeShaderTarget, opts)
		);
		return &it->second;
	}

	// --- BUFFER ARCHITECTURAL FACTORIES ---

	BufferResource* CreateVertexBuffer(const std::string& name, size_t size) {
		auto [it, success] = m_buffers.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, size, BufferUsagePurpose::VertexInput)
		);
		return &it->second;
	}

	BufferResource* CreateUniformBuffer(const std::string& name, size_t size) {
		auto [it, success] = m_buffers.emplace(
			std::piecewise_construct,
			std::forward_as_tuple(name),
			std::forward_as_tuple(m_device, m_allocator, size, BufferUsagePurpose::UniformBlock)
		);
		return &it->second;
	}

	// Global cleanups
	void ClearAllResources() {
		m_textures.clear();
		m_buffers.clear();
	}

	TextureResource* GetTexture(const std::string& name) { return &m_textures.at(name); }

	BufferResource* GetBuffer(const std::string& name) { return &m_buffers.at(name); }

private:
	vk::Device                                       m_device;
	VmaAllocator                                     m_allocator;
	std::unordered_map<std::string, TextureResource> m_textures;
	std::unordered_map<std::string, BufferResource>  m_buffers;
};

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
