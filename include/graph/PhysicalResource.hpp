#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/ResourceState.hpp"
#include "graph/VulkanSeam.hpp"

// The physical layer's Vulkan-owning resource types. Everything here is real vk:: -- this file
// (and everything downstream of it: PhysicalRegistry.hpp, PhysicalExecutionBackend.hpp,
// BarrierTranslator.hpp) requires a real Vulkan SDK to compile. include/graph/Execution.hpp and
// everything below it in the declarative layer stays Vulkan-free; this is the one seam where
// the graph's opaque ResourceDesc::formatCode/usageMask get interpreted as real vk::Format /
// vk::ImageUsageFlags.

namespace brassica::graph {

	namespace detail {

		inline vk::ImageCreateInfo BuildImageCreateInfo(const ResourceDesc& desc) {
			const bool is3D = desc.depth > 1;
			return vk::ImageCreateInfo{
				{},
				is3D ? vk::ImageType::e3D : vk::ImageType::e2D,
				static_cast<vk::Format>(
					desc.formatCode ? desc.formatCode : static_cast<std::uint32_t>(vk::Format::eR8G8B8A8Unorm)
				),
				vk::Extent3D(desc.width ? desc.width : 1, desc.height ? desc.height : 1, desc.depth ? desc.depth : 1),
				desc.mips ? desc.mips : 1,
				desc.layers ? desc.layers : 1,
				vk::SampleCountFlagBits::e1,
				vk::ImageTiling::eOptimal,
				desc.usageMask ? vk::ImageUsageFlags(desc.usageMask)
							   : (vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment),
			};
		}

		inline vk::BufferCreateInfo BuildBufferCreateInfo(const ResourceDesc& desc) {
			return vk::BufferCreateInfo{
				{},
				desc.byteSize ? desc.byteSize : 1,
				desc.usageMask ? vk::BufferUsageFlags(desc.usageMask) : vk::BufferUsageFlagBits::eStorageBuffer,
				vk::SharingMode::eExclusive,
			};
		}

	} // namespace detail

	// Move-only RAII owner of a Vulkan image, in one of three modes:
	//  - Owning: allocates its own memory via VMA. Destructor frees image, view, and memory.
	//  - Imported: wraps an externally-owned image (e.g. a swapchain image). Destructor no-ops
	//    on the Vulkan objects -- this texture never had ownership of them.
	//  - Aliased: bound to a block of memory owned by an ImageAliasPool (see below), not by this
	//    object. Destructor destroys the image and view but leaves the memory alone -- the pool
	//    owns that, and may bind a different image to it later once this one's lifetime ends.
	class PhysicalTexture {
	public:
		enum class Ownership : std::uint8_t { Owning, Imported, Aliased };

		// Owning: fresh allocation.
		PhysicalTexture(vk::Device device, VmaAllocator allocator, const ResourceDesc& desc):
			m_device(device), m_allocator(allocator), m_desc(desc), m_ownership(Ownership::Owning) {
			CreateImage(/*existingAllocation=*/nullptr, /*offsetInBlock=*/0);
			CreateView();
		}

		// Aliased: bound to an existing block, owned by the caller's ImageAliasPool.
		// memoryWasReused marks that the block held a different resource before this one --
		// Vulkan requires a memory dependency against whatever that prior occupant did, and this
		// object has no edge to it (the two are different ResourceIds, so Graph::CollectEdges
		// never connects them). The seed below is deliberately conservative rather than precise:
		// "wait on everything, assume it wrote" is always safe, tracking the exact prior
		// occupant's stage/access is not attempted.
		PhysicalTexture(
			vk::Device          device,
			VmaAllocator        allocator,
			VmaAllocation       aliasedMemory,
			vk::DeviceSize      offsetInBlock,
			const ResourceDesc& desc,
			bool                memoryWasReused = false
		):
			m_device(device),
			m_allocator(allocator),
			m_desc(desc),
			m_ownership(Ownership::Aliased),
			m_lastStage(memoryWasReused ? vk::PipelineStageFlagBits2::eAllCommands : vk::PipelineStageFlags2{}),
			m_lastAccess(
				memoryWasReused ? (vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
								: vk::AccessFlags2{}
			) {
			CreateImage(aliasedMemory, offsetInBlock);
			CreateView();
		}

		// Imported: wraps a texture this object never owned. initialLayout/hasDefinedContents
		// describe what the caller already knows about it. The swapchain re-imports fresh every
		// frame with the defaults (eUndefined, false) -- it's about to be fully overwritten, so
		// discarding prior contents is correct. A long-lived import (e.g. the terrain clipmap)
		// must pass its real state, or every frame's first touch would derive an eUndefined
		// transition and discard it.
		PhysicalTexture(
			vk::Image           image,
			vk::ImageView       view,
			const ResourceDesc& desc,
			vk::ImageLayout     initialLayout = vk::ImageLayout::eUndefined,
			bool                hasDefinedContents = false
		):
			m_image(image),
			m_view(view),
			m_desc(desc),
			m_ownership(Ownership::Imported),
			m_currentLayout(initialLayout),
			m_lastStage(hasDefinedContents ? vk::PipelineStageFlagBits2::eAllCommands : vk::PipelineStageFlags2{}),
			m_lastAccess(
				hasDefinedContents ? (vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
								   : vk::AccessFlags2{}
			),
			m_hasDefinedContents(hasDefinedContents) {}

		~PhysicalTexture() { Cleanup(); }

		PhysicalTexture(const PhysicalTexture&) = delete;
		PhysicalTexture& operator=(const PhysicalTexture&) = delete;

		PhysicalTexture(PhysicalTexture&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_image(o.m_image),
			m_view(o.m_view),
			m_allocation(o.m_allocation),
			m_desc(o.m_desc),
			m_ownership(o.m_ownership),
			m_currentLayout(o.m_currentLayout),
			m_lastStage(o.m_lastStage),
			m_lastAccess(o.m_lastAccess),
			m_hasDefinedContents(o.m_hasDefinedContents) {
			o.m_image = nullptr;
			o.m_view = nullptr;
			o.m_allocation = nullptr;
			o.m_currentLayout = vk::ImageLayout::eUndefined;
			o.m_lastStage = {};
			o.m_lastAccess = {};
			o.m_hasDefinedContents = false;
		}

		PhysicalTexture& operator=(PhysicalTexture&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_image = o.m_image;
				m_view = o.m_view;
				m_allocation = o.m_allocation;
				m_desc = o.m_desc;
				m_ownership = o.m_ownership;
				m_currentLayout = o.m_currentLayout;
				m_lastStage = o.m_lastStage;
				m_lastAccess = o.m_lastAccess;
				m_hasDefinedContents = o.m_hasDefinedContents;

				o.m_image = nullptr;
				o.m_view = nullptr;
				o.m_allocation = nullptr;
				o.m_currentLayout = vk::ImageLayout::eUndefined;
				o.m_lastStage = {};
				o.m_lastAccess = {};
				o.m_hasDefinedContents = false;
			}
			return *this;
		}

		[[nodiscard]] vk::Image GetImage() const { return m_image; }

		[[nodiscard]] vk::ImageView GetView() const { return m_view; }

		[[nodiscard]] const ResourceDesc& GetDesc() const { return m_desc; }

		[[nodiscard]] bool IsImported() const { return m_ownership == Ownership::Imported; }

		[[nodiscard]] bool IsAliased() const { return m_ownership == Ownership::Aliased; }

		// Tracked GPU-side state, mutated only by the barrier-synthesis seam (BarrierTranslator,
		// DynamicRenderingWrapper -- see PhysicalExecutionBackend.hpp/BarrierTranslator.hpp).
		// Kept on the object rather than in a side map so it resets iff the object itself is
		// recreated: a fresh Owning/Imported construction or a freshly-acquired Aliased slot
		// naturally starts from a correct state with no explicit invalidation step required.
		[[nodiscard]] vk::ImageLayout GetCurrentLayout() const { return m_currentLayout; }

		void SetCurrentLayout(vk::ImageLayout layout) { m_currentLayout = layout; }

		[[nodiscard]] vk::PipelineStageFlags2 GetLastStage() const { return m_lastStage; }

		[[nodiscard]] vk::AccessFlags2 GetLastAccess() const { return m_lastAccess; }

		// Barrier tracking updates stage/access together (there is never a reason to change one
		// without the other), so one setter rather than two keeps them from drifting apart.
		void SetLastStageAccess(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
			m_lastStage = stage;
			m_lastAccess = access;
		}

		// Whether this resource, as this key, has meaningful content to preserve. False for a
		// resource nothing has written yet -- the signal DynamicRenderingWrapper::Begin uses to
		// force AttachmentLoadOp::eClear rather than eLoad even for a ReadWrite access, since
		// there is provably nothing to load.
		[[nodiscard]] bool HasDefinedContents() const { return m_hasDefinedContents; }

		void SetHasDefinedContents(bool defined) { m_hasDefinedContents = defined; }

	private:
		void CreateImage(VmaAllocation existingAllocation, vk::DeviceSize offsetInBlock) {
			vk::ImageCreateInfo imageInfo = detail::BuildImageCreateInfo(m_desc);
			VkImageCreateInfo   cImageInfo = static_cast<VkImageCreateInfo>(imageInfo);
			VkImage             rawImage = VK_NULL_HANDLE;

			if (existingAllocation) {
				if (vmaCreateAliasingImage2(m_allocator, existingAllocation, offsetInBlock, &cImageInfo, &rawImage) !=
				    VK_SUCCESS) {
					throw std::runtime_error("Failed to bind aliased image via VMA!");
				}
				m_allocation = existingAllocation; // not owned; see class comment
			} else {
				VmaAllocationCreateInfo allocInfo{};
				allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

				if (vmaCreateImage(m_allocator, &cImageInfo, &allocInfo, &rawImage, &m_allocation, nullptr) !=
				    VK_SUCCESS) {
					throw std::runtime_error("Failed to allocate texture image via VMA!");
				}
			}
			m_image = rawImage;
		}

		void CreateView() {
			const bool is3D = m_desc.depth > 1;
			const auto format = static_cast<vk::Format>(
				m_desc.formatCode ? m_desc.formatCode : static_cast<std::uint32_t>(vk::Format::eR8G8B8A8Unorm)
			);
			vk::ImageViewCreateInfo viewInfo{
				{},
				m_image,
				is3D ? vk::ImageViewType::e3D : vk::ImageViewType::e2D,
				format,
				vk::ComponentMapping{},
				vk::ImageSubresourceRange(
					AspectFor(format),
					0,
					m_desc.mips ? m_desc.mips : 1,
					0,
					m_desc.layers ? m_desc.layers : 1
				),
			};
			m_view = m_device.createImageView(viewInfo);
		}

		void Cleanup() {
			if (m_ownership == Ownership::Imported) {
				return;
			}
			if (m_view) {
				m_device.destroyImageView(m_view);
				m_view = nullptr;
			}
			if (!m_image) {
				return;
			}
			if (m_ownership == Ownership::Owning) {
				vmaDestroyImage(m_allocator, static_cast<VkImage>(m_image), m_allocation); // frees image + memory
			} else {
				m_device.destroyImage(m_image); // Aliased: memory belongs to the pool, not this object
			}
			m_image = nullptr;
			m_allocation = nullptr;
		}

		vk::Device    m_device{};
		VmaAllocator  m_allocator = nullptr;
		vk::Image     m_image{};
		vk::ImageView m_view{};
		VmaAllocation m_allocation = nullptr;
		ResourceDesc  m_desc{};
		Ownership     m_ownership = Ownership::Imported;

		// See the accessors above for what each of these means; declaration order here is the
		// order every constructor's member-initializer list must follow.
		vk::ImageLayout         m_currentLayout{vk::ImageLayout::eUndefined};
		vk::PipelineStageFlags2 m_lastStage{};
		vk::AccessFlags2        m_lastAccess{};
		bool                    m_hasDefinedContents{false};
	};

	// Move-only RAII owner of a Vulkan buffer. Same three ownership modes as PhysicalTexture,
	// minus the view.
	class PhysicalBuffer {
	public:
		enum class Ownership : std::uint8_t { Owning, Imported, Aliased };

		// device isn't needed to create a buffer (unlike a texture's view), but is still taken
		// here and stored so the aliased-mode destructor can call m_device.destroyBuffer(...)
		// -- keeps PhysicalTexture and PhysicalBuffer construction symmetric for callers.
		PhysicalBuffer(vk::Device device, VmaAllocator allocator, const ResourceDesc& desc):
			m_device(device), m_allocator(allocator), m_desc(desc), m_ownership(Ownership::Owning) {
			CreateBuffer(nullptr, 0);
		}

		// memoryWasReused: see PhysicalTexture's aliased constructor comment -- same conservative
		// seed, same reason (a memory dependency against a prior occupant that no graph edge
		// connects to this object).
		PhysicalBuffer(
			vk::Device          device,
			VmaAllocator        allocator,
			VmaAllocation       aliasedMemory,
			vk::DeviceSize      offsetInBlock,
			const ResourceDesc& desc,
			bool                memoryWasReused = false
		):
			m_device(device),
			m_allocator(allocator),
			m_desc(desc),
			m_ownership(Ownership::Aliased),
			m_lastStage(memoryWasReused ? vk::PipelineStageFlagBits2::eAllCommands : vk::PipelineStageFlags2{}),
			m_lastAccess(
				memoryWasReused ? (vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
								: vk::AccessFlags2{}
			) {
			CreateBuffer(aliasedMemory, offsetInBlock);
		}

		// hasDefinedContents: see PhysicalTexture's imported constructor comment. No layout
		// concept for a buffer, so this is the only piece of caller-supplied state.
		PhysicalBuffer(vk::Buffer buffer, const ResourceDesc& desc, bool hasDefinedContents = false):
			m_buffer(buffer),
			m_desc(desc),
			m_ownership(Ownership::Imported),
			m_lastStage(hasDefinedContents ? vk::PipelineStageFlagBits2::eAllCommands : vk::PipelineStageFlags2{}),
			m_lastAccess(
				hasDefinedContents ? (vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite)
								   : vk::AccessFlags2{}
			),
			m_hasDefinedContents(hasDefinedContents) {}

		~PhysicalBuffer() { Cleanup(); }

		PhysicalBuffer(const PhysicalBuffer&) = delete;
		PhysicalBuffer& operator=(const PhysicalBuffer&) = delete;

		PhysicalBuffer(PhysicalBuffer&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_buffer(o.m_buffer),
			m_allocation(o.m_allocation),
			m_desc(o.m_desc),
			m_ownership(o.m_ownership),
			m_lastStage(o.m_lastStage),
			m_lastAccess(o.m_lastAccess),
			m_hasDefinedContents(o.m_hasDefinedContents) {
			o.m_buffer = nullptr;
			o.m_allocation = nullptr;
			o.m_lastStage = {};
			o.m_lastAccess = {};
			o.m_hasDefinedContents = false;
		}

		PhysicalBuffer& operator=(PhysicalBuffer&& o) noexcept {
			if (this != &o) {
				Cleanup();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_buffer = o.m_buffer;
				m_allocation = o.m_allocation;
				m_desc = o.m_desc;
				m_ownership = o.m_ownership;
				m_lastStage = o.m_lastStage;
				m_lastAccess = o.m_lastAccess;
				m_hasDefinedContents = o.m_hasDefinedContents;

				o.m_buffer = nullptr;
				o.m_allocation = nullptr;
				o.m_lastStage = {};
				o.m_lastAccess = {};
				o.m_hasDefinedContents = false;
			}
			return *this;
		}

		[[nodiscard]] vk::Buffer GetBuffer() const { return m_buffer; }

		[[nodiscard]] const ResourceDesc& GetDesc() const { return m_desc; }

		[[nodiscard]] bool IsImported() const { return m_ownership == Ownership::Imported; }

		[[nodiscard]] bool IsAliased() const { return m_ownership == Ownership::Aliased; }

		// See PhysicalTexture's identically-named accessors -- same tracked-state contract,
		// minus a layout (buffers have none).
		[[nodiscard]] vk::PipelineStageFlags2 GetLastStage() const { return m_lastStage; }

		[[nodiscard]] vk::AccessFlags2 GetLastAccess() const { return m_lastAccess; }

		void SetLastStageAccess(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
			m_lastStage = stage;
			m_lastAccess = access;
		}

		[[nodiscard]] bool HasDefinedContents() const { return m_hasDefinedContents; }

		void SetHasDefinedContents(bool defined) { m_hasDefinedContents = defined; }

	private:
		void CreateBuffer(VmaAllocation existingAllocation, vk::DeviceSize offsetInBlock) {
			vk::BufferCreateInfo bufferInfo = detail::BuildBufferCreateInfo(m_desc);

			VkBufferCreateInfo cBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);
			VkBuffer           rawBuffer = VK_NULL_HANDLE;

			if (existingAllocation) {
				if (vmaCreateAliasingBuffer2(
						m_allocator,
						existingAllocation,
						offsetInBlock,
						&cBufferInfo,
						&rawBuffer
					) != VK_SUCCESS) {
					throw std::runtime_error("Failed to bind aliased buffer via VMA!");
				}
				m_allocation = existingAllocation; // not owned; see class comment
			} else {
				VmaAllocationCreateInfo allocInfo{};
				allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

				if (vmaCreateBuffer(m_allocator, &cBufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr) !=
				    VK_SUCCESS) {
					throw std::runtime_error("Failed to allocate buffer via VMA!");
				}
			}
			m_buffer = rawBuffer;
		}

		void Cleanup() {
			if (m_ownership == Ownership::Imported || !m_buffer) {
				return;
			}
			if (m_ownership == Ownership::Owning) {
				vmaDestroyBuffer(m_allocator, static_cast<VkBuffer>(m_buffer), m_allocation);
			} else {
				// Aliased: memory belongs to the pool, not this object -- destroy the buffer
				// object only, leave the memory alone.
				m_device.destroyBuffer(m_buffer);
			}
			m_buffer = nullptr;
			m_allocation = nullptr;
		}

		vk::Device    m_device{};
		VmaAllocator  m_allocator = nullptr;
		vk::Buffer    m_buffer{};
		VmaAllocation m_allocation = nullptr;
		ResourceDesc  m_desc{};
		Ownership     m_ownership = Ownership::Imported;

		vk::PipelineStageFlags2 m_lastStage{};
		vk::AccessFlags2        m_lastAccess{};
		bool                    m_hasDefinedContents{false};
	};

	// Deliberately Imported-only: no Owning/Aliased modes, no VMA, no destructor logic. The
	// graph never builds an acceleration structure itself -- TerrainPass's BLAS/TLAS build stays
	// exactly as it is (its own transient command pool/queue, its own throttle, its own
	// synchronous device.waitIdle()), entirely out-of-band from the graph. This type only wraps
	// the *result* so a graph node can declare a real Read dependency on it and get a real
	// barrier -- adding Owning/Aliased modes here would be dead code with no caller. No
	// hasDefinedContents either: that concept only matters to DynamicRenderingWrapper::Begin,
	// which already skips any realization whose desc.kind isn't Image2D/Image3D, so an AS
	// realization is filtered out there for free.
	//
	// Per the Vulkan spec, acceleration structures are synchronized via a generic, handle-free
	// vk::MemoryBarrier2 -- not an image- or buffer-specific barrier struct (see
	// BarrierTranslator.hpp's AS branch). That's why this type carries no VMA allocation or
	// destroy path even conceptually: nothing downstream of this type ever dereferences the
	// handle for anything beyond storing/returning it.
	class PhysicalAccelerationStructure {
	public:
		explicit PhysicalAccelerationStructure(vk::AccelerationStructureKHR as): m_as(as) {}

		[[nodiscard]] vk::AccelerationStructureKHR Get() const { return m_as; }

		[[nodiscard]] vk::PipelineStageFlags2 GetLastStage() const { return m_lastStage; }

		[[nodiscard]] vk::AccessFlags2 GetLastAccess() const { return m_lastAccess; }

		void SetLastStageAccess(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
			m_lastStage = stage;
			m_lastAccess = access;
		}

	private:
		vk::AccelerationStructureKHR m_as{};
		vk::PipelineStageFlags2      m_lastStage{};
		vk::AccessFlags2             m_lastAccess{};
	};

	// Reuses one VmaAllocation across a sequence of images whose stage lifetimes don't overlap,
	// via VMA's aliasing API (vmaCreateAliasingImage2) rather than allocating fresh memory every
	// time. Greedy first-fit -- not a globally-optimal bin-packing, which is NP-hard in general.
	// Sizing uses Vulkan 1.3 core's vkGetDeviceImageMemoryRequirements (wrapped here as
	// vk::Device::getImageMemoryRequirements(DeviceImageMemoryRequirements)), which reports the
	// memory a resource would need from just its create-info, with no image object required.
	class ImageAliasPool {
	public:
		ImageAliasPool(vk::Device device, VmaAllocator allocator): m_device(device), m_allocator(allocator) {}

		// Binds a new image to a reused block if one whose occupant's lifetime has fully ended
		// is big enough, otherwise allocates a fresh block sized to this resource.
		std::shared_ptr<PhysicalTexture>
		Acquire(const ResourceDesc& desc, std::size_t firstStage, std::size_t lastStage) {
			vk::ImageCreateInfo  imageInfo = detail::BuildImageCreateInfo(desc);
			const vk::DeviceSize requiredSize = m_device
													.getImageMemoryRequirements(
														vk::DeviceImageMemoryRequirements{&imageInfo}
													)
													.memoryRequirements.size;

			for (auto& block : m_blocks) {
				if (block.lastOccupantStage < firstStage && block.size >= requiredSize) {
					block.lastOccupantStage = lastStage;
					// memoryWasReused=true: this block held a different resource before this
					// one, so the new PhysicalTexture seeds a conservative "wait on everything"
					// state -- see the constructor comment.
					return std::make_shared<PhysicalTexture>(m_device, m_allocator, block.allocation, 0, desc, true);
				}
			}

			// VMA_MEMORY_USAGE_AUTO* only works through vmaCreateImage/vmaCreateBuffer (or their
			// ForImage/ForBuffer variants) -- those pass VMA the full create-info so its
			// heuristic can pick a memory type. Plain vmaAllocateMemory only ever sees
			// VkMemoryRequirements, so AUTO can't be resolved there and the call fails outright
			// (this is a documented VMA constraint, not a resource-limit failure -- confirmed by
			// it failing immediately on a real device with plenty of free memory). Fall back to
			// the pre-3.0-style explicit request instead.
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_UNKNOWN;
			allocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(vk::MemoryPropertyFlagBits::eDeviceLocal);
			VkMemoryRequirements cRequirements = static_cast<VkMemoryRequirements>(
				m_device.getImageMemoryRequirements(vk::DeviceImageMemoryRequirements{&imageInfo}).memoryRequirements
			);

			VmaAllocation allocation = nullptr;
			if (vmaAllocateMemory(m_allocator, &cRequirements, &allocInfo, &allocation, nullptr) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate aliasing memory block via VMA!");
			}
			m_blocks.push_back(Block{allocation, requiredSize, lastStage});
			return std::make_shared<PhysicalTexture>(m_device, m_allocator, allocation, 0, desc, false);
		}

		// Frees every block's memory. Safe to call only once nothing still references a
		// PhysicalTexture bound to one of these blocks -- those destructors only destroy the
		// image/view, never the memory, so this is the one place the memory itself is freed.
		void Reset() {
			for (auto& block : m_blocks) {
				if (block.allocation) {
					vmaFreeMemory(m_allocator, block.allocation);
				}
			}
			m_blocks.clear();
		}

		~ImageAliasPool() { Reset(); }

		ImageAliasPool(const ImageAliasPool&) = delete;
		ImageAliasPool& operator=(const ImageAliasPool&) = delete;
		ImageAliasPool(ImageAliasPool&&) = default;
		ImageAliasPool& operator=(ImageAliasPool&&) = default;

	private:
		struct Block {
			VmaAllocation  allocation = nullptr;
			vk::DeviceSize size = 0;
			std::size_t    lastOccupantStage = 0;
		};

		vk::Device         m_device{};
		VmaAllocator       m_allocator = nullptr;
		std::vector<Block> m_blocks;
	};

	// Buffer counterpart of ImageAliasPool. Kept as a separate pool rather than sharing blocks
	// with images -- simpler, at the cost of being less maximally space-efficient; that's the
	// one place this aliasing scheme is deliberately not the most aggressive it could be.
	class BufferAliasPool {
	public:
		BufferAliasPool(vk::Device device, VmaAllocator allocator): m_device(device), m_allocator(allocator) {}

		std::shared_ptr<PhysicalBuffer>
		Acquire(const ResourceDesc& desc, std::size_t firstStage, std::size_t lastStage) {
			vk::BufferCreateInfo bufferInfo = detail::BuildBufferCreateInfo(desc);
			const vk::DeviceSize requiredSize = m_device
													.getBufferMemoryRequirements(
														vk::DeviceBufferMemoryRequirements{&bufferInfo}
													)
													.memoryRequirements.size;

			for (auto& block : m_blocks) {
				if (block.lastOccupantStage < firstStage && block.size >= requiredSize) {
					block.lastOccupantStage = lastStage;
					return std::make_shared<PhysicalBuffer>(m_device, m_allocator, block.allocation, 0, desc, true);
				}
			}

			// Same VMA constraint as ImageAliasPool::Acquire: VMA_MEMORY_USAGE_AUTO* only
			// resolves through vmaCreateBuffer/vmaAllocateMemoryForBuffer, which see the full
			// create-info; plain vmaAllocateMemory only sees VkMemoryRequirements and fails
			// outright with AUTO.
			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_UNKNOWN;
			allocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(vk::MemoryPropertyFlagBits::eDeviceLocal);
			VkMemoryRequirements cRequirements = static_cast<VkMemoryRequirements>(
				m_device.getBufferMemoryRequirements(vk::DeviceBufferMemoryRequirements{&bufferInfo}).memoryRequirements
			);

			VmaAllocation allocation = nullptr;
			if (vmaAllocateMemory(m_allocator, &cRequirements, &allocInfo, &allocation, nullptr) != VK_SUCCESS) {
				throw std::runtime_error("Failed to allocate aliasing memory block via VMA!");
			}
			m_blocks.push_back(Block{allocation, requiredSize, lastStage});
			return std::make_shared<PhysicalBuffer>(m_device, m_allocator, allocation, 0, desc, false);
		}

		void Reset() {
			for (auto& block : m_blocks) {
				if (block.allocation) {
					vmaFreeMemory(m_allocator, block.allocation);
				}
			}
			m_blocks.clear();
		}

		~BufferAliasPool() { Reset(); }

		BufferAliasPool(const BufferAliasPool&) = delete;
		BufferAliasPool& operator=(const BufferAliasPool&) = delete;
		BufferAliasPool(BufferAliasPool&&) = default;
		BufferAliasPool& operator=(BufferAliasPool&&) = default;

	private:
		struct Block {
			VmaAllocation  allocation = nullptr;
			vk::DeviceSize size = 0;
			std::size_t    lastOccupantStage = 0;
		};

		vk::Device         m_device{};
		VmaAllocator       m_allocator = nullptr;
		std::vector<Block> m_blocks;
	};

	// -- Preset helpers -------------------------------------------------------------------
	// Ergonomic ResourceDesc construction for common purposes, mirroring the old Resources.hpp
	// ImageUsagePurpose/BufferUsagePurpose cases -- but producing the graph's own ResourceDesc
	// rather than a competing options struct. A node's Setup() calls one of these to fill in
	// Recipe::realizations legibly.

	inline ResourceDesc ColorAttachmentDesc(
		std::uint32_t width,
		std::uint32_t height,
		vk::Format    format = vk::Format::eR16G16B16A16Sfloat
	) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.formatCode = static_cast<std::uint32_t>(format),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
			),
		};
	}

	inline ResourceDesc ShadingRateAttachmentDesc(
		std::uint32_t width,
		std::uint32_t height,
		vk::Format    format = vk::Format::eR8Uint
	) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.formatCode = static_cast<std::uint32_t>(format),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR |
				vk::ImageUsageFlagBits::eStorage |
				vk::ImageUsageFlagBits::eTransferDst
			),
		};
	}

	inline ResourceDesc
	DepthBufferDesc(std::uint32_t width, std::uint32_t height, vk::Format format = vk::Format::eD32Sfloat) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.formatCode = static_cast<std::uint32_t>(format),
			.usageMask = static_cast<std::uint32_t>(vk::ImageUsageFlagBits::eDepthStencilAttachment),
		};
	}

	inline ResourceDesc ComputeStorageImageDesc(
		std::uint32_t width,
		std::uint32_t height,
		vk::Format    format = vk::Format::eR16G16B16A16Sfloat
	) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Image2D,
			.width = width,
			.height = height,
			.formatCode = static_cast<std::uint32_t>(format),
			.usageMask = static_cast<std::uint32_t>(
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
			),
		};
	}

	inline ResourceDesc UniformBufferDesc(std::uint64_t byteSize) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Buffer,
			.usageMask = static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eUniformBuffer),
			.byteSize = byteSize,
		};
	}

	inline ResourceDesc StorageBufferDesc(std::uint64_t byteSize) {
		return ResourceDesc{
			.kind = ResourceDesc::Kind::Buffer,
			.usageMask = static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eStorageBuffer),
			.byteSize = byteSize,
		};
	}

	// No extent/format/usage is meaningful for an acceleration structure -- see
	// PhysicalAccelerationStructure's class comment for why this kind carries no other state.
	inline ResourceDesc AccelerationStructureDesc() {
		return ResourceDesc{.kind = ResourceDesc::Kind::AccelerationStructure};
	}

} // namespace brassica::graph
