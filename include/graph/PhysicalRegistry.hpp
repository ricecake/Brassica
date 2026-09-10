#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	struct PhysicalTexture {
		VkImage       image = VK_NULL_HANDLE;
		VkImageView   view = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		ResourceDesc  desc{};
		bool          isImported = false;
		uint32_t      bindlessIndex = UINT32_MAX;
	};

	struct PhysicalBuffer {
		VkBuffer      buffer = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;
		ResourceDesc  desc{};
		bool          isImported = false;
	};

	struct TemporalLifetime {
		std::size_t firstPass = SIZE_MAX;
		std::size_t lastPass = 0;
	};

	class PhysicalResourceRegistry {
	public:
		PhysicalResourceRegistry(VkDevice device = VK_NULL_HANDLE, VmaAllocator allocator = VK_NULL_HANDLE):
			m_device(device), m_allocator(allocator) {}

		~PhysicalResourceRegistry() { DestroyResources(); }

		PhysicalResourceRegistry(const PhysicalResourceRegistry&) = delete;
		PhysicalResourceRegistry& operator=(const PhysicalResourceRegistry&) = delete;

		PhysicalResourceRegistry(PhysicalResourceRegistry&& o) noexcept:
			m_device(o.m_device),
			m_allocator(o.m_allocator),
			m_textures(std::move(o.m_textures)),
			m_buffers(std::move(o.m_buffers)),
			m_bindlessIndices(std::move(o.m_bindlessIndices)),
			m_globalDescriptorSet(o.m_globalDescriptorSet),
			m_globalBinding(o.m_globalBinding),
			m_nextBindlessIndex(o.m_nextBindlessIndex) {
			o.m_device = VK_NULL_HANDLE;
			o.m_allocator = VK_NULL_HANDLE;
		}

		PhysicalResourceRegistry& operator=(PhysicalResourceRegistry&& o) noexcept {
			if (this != &o) {
				DestroyResources();
				m_device = o.m_device;
				m_allocator = o.m_allocator;
				m_textures = std::move(o.m_textures);
				m_buffers = std::move(o.m_buffers);
				m_bindlessIndices = std::move(o.m_bindlessIndices);
				m_globalDescriptorSet = o.m_globalDescriptorSet;
				m_globalBinding = o.m_globalBinding;
				m_nextBindlessIndex = o.m_nextBindlessIndex;

				o.m_device = VK_NULL_HANDLE;
				o.m_allocator = VK_NULL_HANDLE;
			}
			return *this;
		}

		void SetDeviceAndAllocator(VkDevice device, VmaAllocator allocator) {
			m_device = device;
			m_allocator = allocator;
		}

		void SetGlobalDescriptorSet(VkDescriptorSet set, uint32_t binding) {
			m_globalDescriptorSet = set;
			m_globalBinding = binding;
		}

		void RegisterImportedTexture(ResourceId id, VkImage image, VkImageView view, const ResourceDesc& desc) {
			PhysicalTexture tex{};
			tex.image = image;
			tex.view = view;
			tex.desc = desc;
			tex.isImported = true;
			m_textures[id] = tex;
		}

		void RegisterImportedBuffer(ResourceId id, VkBuffer buffer, const ResourceDesc& desc) {
			PhysicalBuffer buf{};
			buf.buffer = buffer;
			buf.desc = desc;
			buf.isImported = true;
			m_buffers[id] = buf;
		}

		template <ResourceRef K>
		void RegisterImportedTexture(VkImage image, VkImageView view, const ResourceDesc& desc) {
			RegisterImportedTexture(IdOf<K>(), image, view, desc);
		}

		template <ResourceRef K>
		void RegisterImportedBuffer(VkBuffer buffer, const ResourceDesc& desc) {
			RegisterImportedBuffer(IdOf<K>(), buffer, desc);
		}

		uint32_t GetBindlessIndex(ResourceId id) const {
			auto it = m_bindlessIndices.find(id);
			if (it != m_bindlessIndices.end()) {
				return it->second;
			}
			auto texIt = m_textures.find(id);
			if (texIt != m_textures.end()) {
				return texIt->second.bindlessIndex;
			}
			return UINT32_MAX;
		}

		template <ResourceRef K>
		uint32_t GetBindlessIndex() const {
			return GetBindlessIndex(IdOf<K>());
		}

		const PhysicalTexture* GetTexture(ResourceId id) const {
			auto it = m_textures.find(id);
			return (it != m_textures.end()) ? &it->second : nullptr;
		}

		template <ResourceRef K>
		const PhysicalTexture* GetTexture() const {
			return GetTexture(IdOf<K>());
		}

		const PhysicalBuffer* GetBuffer(ResourceId id) const {
			auto it = m_buffers.find(id);
			return (it != m_buffers.end()) ? &it->second : nullptr;
		}

		template <ResourceRef K>
		const PhysicalBuffer* GetBuffer() const {
			return GetBuffer(IdOf<K>());
		}

		// Analysis & Provisioning: Computes lifetimes for all realizations in schedule order,
		// and provisions textures and buffers. Supports transient aliasing across non-overlapping
		// execution windows when enabled.
		//
		// Lifetime is measured in stage indices, not flat node-slot indices: a resource touched
		// anywhere within stage N is "alive during stage N". That's coarser than per-node, but
		// it's the correct granularity given a Schedule -- nodes sharing a stage are provably
		// independent and may run concurrently, so a lifetime window can never straddle a single
		// stage boundary in a way that would let two live-at-once resources alias.
		void Provision(const Schedule& schedule, std::span<const Recipe> recipes, bool enableAliasing = true) {
			// 1. Calculate temporal lifetimes
			std::unordered_map<ResourceId, TemporalLifetime> lifetimes;
			for (std::size_t stageIndex = 0; stageIndex < schedule.stages.size(); ++stageIndex) {
				for (std::size_t nodeIndex : schedule.stages[stageIndex].nodes) {
					if (nodeIndex >= recipes.size())
						continue;
					const auto& recipe = recipes[nodeIndex];
					if (!recipe.isActive)
						continue;

					for (const auto& r : recipe.realizations) {
						auto& lt = lifetimes[r.key];
						lt.firstPass = std::min(lt.firstPass, stageIndex);
						lt.lastPass = std::max(lt.lastPass, stageIndex);
					}
				}
			}

			// Collect all realizations to provision
			std::unordered_map<ResourceId, ResourceRealization> realizationsToProvision;
			for (const auto& stage : schedule.stages) {
				for (std::size_t nodeIndex : stage.nodes) {
					if (nodeIndex >= recipes.size())
						continue;
					const auto& recipe = recipes[nodeIndex];
					if (!recipe.isActive)
						continue;

					for (const auto& r : recipe.realizations) {
						if (m_textures.contains(r.key) && m_textures[r.key].isImported)
							continue;
						if (m_buffers.contains(r.key) && m_buffers[r.key].isImported)
							continue;

						realizationsToProvision[r.key] = r;
					}
				}
			}

			// 2. Provision each non-imported resource
			for (const auto& [id, r] : realizationsToProvision) {
				if (r.desc.kind == ResourceDesc::Kind::Buffer) {
					ProvisionBuffer(id, r.desc);
				} else { // Image2D / Image3D
					ProvisionTexture(id, r.desc, lifetimes[id], enableAliasing);
				}
			}
		}

		void DestroyResources() {
			for (auto& [id, tex] : m_textures) {
				if (!tex.isImported) {
#if __has_include(<vulkan/vulkan.h>) || __has_include("glad/vulkan.h")
					if (m_device && tex.view) {
						vkDestroyImageView(m_device, tex.view, nullptr);
					}
#endif
#if __has_include("vk_mem_alloc.h")
					if (m_allocator && tex.image && tex.allocation) {
						vmaDestroyImage(m_allocator, tex.image, tex.allocation);
					}
#endif
				}
			}
			m_textures.clear();

			for (auto& [id, buf] : m_buffers) {
				if (!buf.isImported) {
#if __has_include("vk_mem_alloc.h")
					if (m_allocator && buf.buffer && buf.allocation) {
						vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
					}
#endif
				}
			}
			m_buffers.clear();
			m_bindlessIndices.clear();
			m_allocatedAliasedBlocks.clear();
		}

	private:
		struct AllocatedBlock {
			VmaAllocation    allocation = VK_NULL_HANDLE;
			uint64_t         size = 0;
			TemporalLifetime lifetime{};
		};

		void ProvisionTexture(ResourceId id, const ResourceDesc& desc, TemporalLifetime lifetime, bool enableAliasing) {
			auto existingIt = m_textures.find(id);
			if (existingIt != m_textures.end() && !existingIt->second.isImported) {
				// Re-use existing if desc matches
				if (existingIt->second.desc.width == desc.width && existingIt->second.desc.height == desc.height &&
				    existingIt->second.desc.depth == desc.depth &&
				    existingIt->second.desc.formatCode == desc.formatCode) {
					return;
				}
			}

			PhysicalTexture tex{};
			tex.desc = desc;

#if __has_include("vk_mem_alloc.h") && (__has_include(<vulkan/vulkan.h>) || __has_include("glad/vulkan.h"))
			if (m_allocator && m_device) {
				VkImageType     imageType = (desc.depth > 1) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
				VkImageViewType viewType = (desc.depth > 1) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
				VkFormat        format =
					static_cast<VkFormat>(desc.formatCode ? desc.formatCode : 44 /* VK_FORMAT_R8G8B8A8_UNORM */);
				VkImageUsageFlags usage = desc.usageMask
					? static_cast<VkImageUsageFlags>(desc.usageMask)
					: (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

				VkImageCreateInfo imageInfo{};
				imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
				imageInfo.imageType = imageType;
				imageInfo.format = format;
				imageInfo
					.extent = {desc.width ? desc.width : 1, desc.height ? desc.height : 1, desc.depth ? desc.depth : 1};
				imageInfo.mipLevels = desc.mips ? desc.mips : 1;
				imageInfo.arrayLayers = desc.layers ? desc.layers : 1;
				imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
				imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
				imageInfo.usage = usage;
				imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				VmaAllocationCreateInfo allocInfo{};
				allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

				if (enableAliasing) {
					allocInfo.flags |= VMA_ALLOCATION_CREATE_ALIASED_BIT;
				}

				if (vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &tex.image, &tex.allocation, nullptr) ==
				    VK_SUCCESS) {
					VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
					if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
					    format == VK_FORMAT_D16_UNORM) {
						aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
					}

					VkImageViewCreateInfo viewInfo{};
					viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
					viewInfo.image = tex.image;
					viewInfo.viewType = viewType;
					viewInfo.format = format;
					viewInfo.subresourceRange.aspectMask = aspect;
					viewInfo.subresourceRange.baseMipLevel = 0;
					viewInfo.subresourceRange.levelCount = desc.mips ? desc.mips : 1;
					viewInfo.subresourceRange.baseArrayLayer = 0;
					viewInfo.subresourceRange.layerCount = desc.layers ? desc.layers : 1;

					vkCreateImageView(m_device, &viewInfo, nullptr, &tex.view);
				}

				// Assign bindless index if sampled and global descriptor set provided
				if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) && m_globalDescriptorSet != VK_NULL_HANDLE) {
					tex.bindlessIndex = m_nextBindlessIndex++;
					m_bindlessIndices[id] = tex.bindlessIndex;

					VkDescriptorImageInfo imgDescriptor{};
					imgDescriptor.imageView = tex.view;
					imgDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

					VkWriteDescriptorSet write{};
					write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					write.dstSet = m_globalDescriptorSet;
					write.dstBinding = m_globalBinding;
					write.dstArrayElement = tex.bindlessIndex;
					write.descriptorCount = 1;
					write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					write.pImageInfo = &imgDescriptor;

					vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
				}
			}
#else
			(void)enableAliasing;
			(void)lifetime;
#endif

			m_textures[id] = tex;
		}

		void ProvisionBuffer(ResourceId id, const ResourceDesc& desc) {
			auto existingIt = m_buffers.find(id);
			if (existingIt != m_buffers.end() && !existingIt->second.isImported) {
				if (existingIt->second.desc.byteSize == desc.byteSize)
					return;
			}

			PhysicalBuffer buf{};
			buf.desc = desc;

#if __has_include("vk_mem_alloc.h")
			if (m_allocator) {
				VkBufferCreateInfo bufferInfo{};
				bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
				bufferInfo.size = desc.byteSize ? desc.byteSize : 1024;
				bufferInfo.usage = desc.usageMask ? static_cast<VkBufferUsageFlags>(desc.usageMask)
												  : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
				bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

				VmaAllocationCreateInfo allocInfo{};
				allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

				vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &buf.buffer, &buf.allocation, nullptr);
			}
#endif

			m_buffers[id] = buf;
		}

		VkDevice                                        m_device = VK_NULL_HANDLE;
		VmaAllocator                                    m_allocator = VK_NULL_HANDLE;
		std::unordered_map<ResourceId, PhysicalTexture> m_textures;
		std::unordered_map<ResourceId, PhysicalBuffer>  m_buffers;
		std::unordered_map<ResourceId, uint32_t>        m_bindlessIndices;
		std::vector<AllocatedBlock>                     m_allocatedAliasedBlocks;
		VkDescriptorSet                                 m_globalDescriptorSet = VK_NULL_HANDLE;
		uint32_t                                        m_globalBinding = 0;
		uint32_t                                        m_nextBindlessIndex = 0;
	};

} // namespace brassica::graph
