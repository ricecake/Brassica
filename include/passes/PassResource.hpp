#pragma once

#include <cstdint>
#include "vulkan/vulkan.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	struct RenderContext {
		vk::CommandBuffer commandBuffer{nullptr};
		VmaAllocator      allocator{VK_NULL_HANDLE};
		vk::Device         device{nullptr};
	};

	// Helper usage flags encoding stages and layout/access for preRead/preWrite
	enum class TextureUsage : uint32_t {
		Undefined = 0,
		ColorAttachment = 1 << 0,
		DepthStencilAttachment = 1 << 1,
		SampledShaderRead = 1 << 2,
		StorageRead = 1 << 3,
		StorageWrite = 1 << 4,
		TransferSrc = 1 << 5,
		TransferDst = 1 << 6,
		Present = 1 << 7
	};

	inline TextureUsage operator|(TextureUsage a, TextureUsage b) {
		return static_cast<TextureUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	inline bool operator&(TextureUsage a, TextureUsage b) {
		return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
	}

	enum class BufferUsage : uint32_t {
		Undefined = 0,
		UniformBuffer = 1 << 0,
		StorageRead = 1 << 1,
		StorageWrite = 1 << 2,
		TransferSrc = 1 << 3,
		TransferDst = 1 << 4
	};

	inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
		return static_cast<BufferUsage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	inline bool operator&(BufferUsage a, BufferUsage b) {
		return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
	}

	// Base resource handle wrapper
	struct PassResource {
		virtual ~PassResource() = default;
	};

	// 2D Texture Resource for FrameGraph
	struct FrameGraphTexture2D : public PassResource {
		struct Desc {
			vk::Extent2D        extent{0, 0};
			vk::Format          format{vk::Format::eUndefined};
			vk::ImageUsageFlags usage{vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled};
		};

		vk::Image       image{nullptr};
		vk::ImageView   imageView{nullptr};
		VmaAllocation   allocation{VK_NULL_HANDLE};
		vk::ImageLayout currentLayout{vk::ImageLayout::eUndefined};

		FrameGraphTexture2D() = default;
		FrameGraphTexture2D(vk::Image img, vk::ImageView view, vk::ImageLayout layout = vk::ImageLayout::eUndefined)
			: image(img), imageView(view), currentLayout(layout) {}

		FrameGraphTexture2D(FrameGraphTexture2D&& o) noexcept
			: image(o.image), imageView(o.imageView), allocation(o.allocation), currentLayout(o.currentLayout) {
			o.image = nullptr;
			o.imageView = nullptr;
			o.allocation = VK_NULL_HANDLE;
		}

		FrameGraphTexture2D& operator=(FrameGraphTexture2D&& o) noexcept {
			if (this != &o) {
				image = o.image;
				imageView = o.imageView;
				allocation = o.allocation;
				currentLayout = o.currentLayout;
				o.image = nullptr;
				o.imageView = nullptr;
				o.allocation = VK_NULL_HANDLE;
			}
			return *this;
		}

		void create(const Desc& desc, void* context);
		void destroy(const Desc& desc, void* context);

		void preRead(const Desc& desc, uint32_t flags, void* context);
		void preWrite(const Desc& desc, uint32_t flags, void* context);
	};

	// 3D Texture Resource for FrameGraph
	struct FrameGraphTexture3D : public PassResource {
		struct Desc {
			vk::Extent3D        extent{0, 0, 0};
			vk::Format          format{vk::Format::eUndefined};
			vk::ImageUsageFlags usage{vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled};
		};

		vk::Image       image{nullptr};
		vk::ImageView   imageView{nullptr};
		VmaAllocation   allocation{VK_NULL_HANDLE};
		vk::ImageLayout currentLayout{vk::ImageLayout::eUndefined};

		FrameGraphTexture3D() = default;
		FrameGraphTexture3D(vk::Image img, vk::ImageView view, vk::ImageLayout layout = vk::ImageLayout::eUndefined)
			: image(img), imageView(view), currentLayout(layout) {}

		FrameGraphTexture3D(FrameGraphTexture3D&& o) noexcept
			: image(o.image), imageView(o.imageView), allocation(o.allocation), currentLayout(o.currentLayout) {
			o.image = nullptr;
			o.imageView = nullptr;
			o.allocation = VK_NULL_HANDLE;
		}

		FrameGraphTexture3D& operator=(FrameGraphTexture3D&& o) noexcept {
			if (this != &o) {
				image = o.image;
				imageView = o.imageView;
				allocation = o.allocation;
				currentLayout = o.currentLayout;
				o.image = nullptr;
				o.imageView = nullptr;
				o.allocation = VK_NULL_HANDLE;
			}
			return *this;
		}

		void create(const Desc& desc, void* context);
		void destroy(const Desc& desc, void* context);

		void preRead(const Desc& desc, uint32_t flags, void* context);
		void preWrite(const Desc& desc, uint32_t flags, void* context);
	};

	// Storage Buffer (SSBO) Resource for FrameGraph
	struct FrameGraphSSBO : public PassResource {
		struct Desc {
			vk::DeviceSize       size{0};
			vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst};
		};

		vk::Buffer    buffer{nullptr};
		VmaAllocation allocation{VK_NULL_HANDLE};
		void*         mappedData{nullptr};

		FrameGraphSSBO() = default;
		FrameGraphSSBO(vk::Buffer buf, VmaAllocation alloc = VK_NULL_HANDLE, void* mapped = nullptr)
			: buffer(buf), allocation(alloc), mappedData(mapped) {}

		FrameGraphSSBO(FrameGraphSSBO&& o) noexcept
			: buffer(o.buffer), allocation(o.allocation), mappedData(o.mappedData) {
			o.buffer = nullptr;
			o.allocation = VK_NULL_HANDLE;
			o.mappedData = nullptr;
		}

		FrameGraphSSBO& operator=(FrameGraphSSBO&& o) noexcept {
			if (this != &o) {
				buffer = o.buffer;
				allocation = o.allocation;
				mappedData = o.mappedData;
				o.buffer = nullptr;
				o.allocation = VK_NULL_HANDLE;
				o.mappedData = nullptr;
			}
			return *this;
		}

		void create(const Desc& desc, void* context);
		void destroy(const Desc& desc, void* context);

		void preRead(const Desc& desc, uint32_t flags, void* context);
		void preWrite(const Desc& desc, uint32_t flags, void* context);
	};

	// Uniform Buffer (UBO) Resource for FrameGraph
	struct FrameGraphUBO : public PassResource {
		struct Desc {
			vk::DeviceSize       size{0};
			vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eUniformBuffer};
		};

		vk::Buffer    buffer{nullptr};
		VmaAllocation allocation{VK_NULL_HANDLE};
		void*         mappedData{nullptr};

		FrameGraphUBO() = default;
		FrameGraphUBO(vk::Buffer buf, VmaAllocation alloc = VK_NULL_HANDLE, void* mapped = nullptr)
			: buffer(buf), allocation(alloc), mappedData(mapped) {}

		FrameGraphUBO(FrameGraphUBO&& o) noexcept
			: buffer(o.buffer), allocation(o.allocation), mappedData(o.mappedData) {
			o.buffer = nullptr;
			o.allocation = VK_NULL_HANDLE;
			o.mappedData = nullptr;
		}

		FrameGraphUBO& operator=(FrameGraphUBO&& o) noexcept {
			if (this != &o) {
				buffer = o.buffer;
				allocation = o.allocation;
				mappedData = o.mappedData;
				o.buffer = nullptr;
				o.allocation = VK_NULL_HANDLE;
				o.mappedData = nullptr;
			}
			return *this;
		}

		void create(const Desc& desc, void* context);
		void destroy(const Desc& desc, void* context);

		void preRead(const Desc& desc, uint32_t flags, void* context);
		void preWrite(const Desc& desc, uint32_t flags, void* context);
	};

	using FrameGraphTexture = FrameGraphTexture2D;

} // namespace brassica
