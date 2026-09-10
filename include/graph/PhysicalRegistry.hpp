#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "graph/Execution.hpp"
#include "graph/Graph.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/ResourceKey.hpp"
#include "graph/ResourceState.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	struct TemporalLifetime {
		std::size_t firstPass = SIZE_MAX;
		std::size_t lastPass = 0;
	};

	// Owns (or references, for imported resources) the physical Vulkan objects backing the
	// graph's declared resource keys, provisioning them from a compiled Schedule. Handles are
	// std::shared_ptr<PhysicalTexture>/<PhysicalBuffer>: a resource handed out to a consumer
	// stays alive until that reference drops, even if Reset() or a later Provision() call has
	// since released the registry's own reference -- real shared ownership, not just a
	// same-frame borrow.
	class PhysicalResourceRegistry {
	public:
		PhysicalResourceRegistry(vk::Device device = {}, VmaAllocator allocator = nullptr):
			m_device(device), m_allocator(allocator), m_imagePool(device, allocator), m_bufferPool(device, allocator) {}

		~PhysicalResourceRegistry() { Reset(); }

		PhysicalResourceRegistry(const PhysicalResourceRegistry&) = delete;
		PhysicalResourceRegistry& operator=(const PhysicalResourceRegistry&) = delete;
		PhysicalResourceRegistry(PhysicalResourceRegistry&&) = default;
		PhysicalResourceRegistry& operator=(PhysicalResourceRegistry&&) = default;

		void SetDeviceAndAllocator(vk::Device device, VmaAllocator allocator) {
			m_device = device;
			m_allocator = allocator;
			m_imagePool = ImageAliasPool(device, allocator);
			m_bufferPool = BufferAliasPool(device, allocator);
		}

		void SetGlobalDescriptorSet(vk::DescriptorSet set, std::uint32_t binding) {
			m_globalDescriptorSet = set;
			m_globalBinding = binding;
		}

		// initialLayout/hasDefinedContents describe what the caller already knows about this
		// import -- see PhysicalTexture's Imported constructor. Defaults are right for a
		// resource re-imported fresh every frame (e.g. the swapchain: about to be fully
		// overwritten, so eUndefined/false is correct); a long-lived import (e.g. the terrain
		// clipmap) must pass its real state or its first touch each frame would derive an
		// eUndefined transition and discard it.
		void RegisterImportedTexture(
			ResourceId          id,
			vk::Image           image,
			vk::ImageView       view,
			const ResourceDesc& desc,
			vk::ImageLayout     initialLayout = vk::ImageLayout::eUndefined,
			bool                hasDefinedContents = false
		) {
			m_textures[id] = std::make_shared<PhysicalTexture>(image, view, desc, initialLayout, hasDefinedContents);
		}

		void RegisterImportedBuffer(
			ResourceId          id,
			vk::Buffer          buffer,
			const ResourceDesc& desc,
			bool                hasDefinedContents = false
		) {
			m_buffers[id] = std::make_shared<PhysicalBuffer>(buffer, desc, hasDefinedContents);
		}

		template <ResourceRef K>
		void RegisterImportedTexture(
			vk::Image           image,
			vk::ImageView       view,
			const ResourceDesc& desc,
			vk::ImageLayout     initialLayout = vk::ImageLayout::eUndefined,
			bool                hasDefinedContents = false
		) {
			RegisterImportedTexture(IdOf<K>(), image, view, desc, initialLayout, hasDefinedContents);
		}

		template <ResourceRef K>
		void RegisterImportedBuffer(vk::Buffer buffer, const ResourceDesc& desc, bool hasDefinedContents = false) {
			RegisterImportedBuffer(IdOf<K>(), buffer, desc, hasDefinedContents);
		}

		// No desc, no initial-state parameters: unlike a texture/buffer import, there is nothing
		// caller-supplied to describe here (see PhysicalAccelerationStructure's class comment,
		// PhysicalResource.hpp) -- the handle itself is the entire state.
		void RegisterImportedAccelerationStructure(ResourceId id, vk::AccelerationStructureKHR as) {
			m_accelStructs[id] = std::make_shared<PhysicalAccelerationStructure>(as);
		}

		template <ResourceRef K>
		void RegisterImportedAccelerationStructure(vk::AccelerationStructureKHR as) {
			RegisterImportedAccelerationStructure(IdOf<K>(), as);
		}

		[[nodiscard]] std::uint32_t GetBindlessIndex(ResourceId id) const {
			auto it = m_bindlessIndices.find(id);
			return it != m_bindlessIndices.end() ? it->second : UINT32_MAX;
		}

		template <ResourceRef K>
		[[nodiscard]] std::uint32_t GetBindlessIndex() const {
			return GetBindlessIndex(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalTexture> GetTexture(ResourceId id) const {
			auto it = m_textures.find(id);
			return it != m_textures.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<const PhysicalTexture> GetTexture() const {
			return GetTexture(IdOf<K>());
		}

		// Non-const overloads, resolved by the constness of the registry reference the caller
		// holds. This is what lets DynamicRenderingWrapper::Begin (const PhysicalResourceRegistry&)
		// read tracked state but never mutate it, while BarrierTranslator (non-const registry)
		// can update layout/stage/access after emitting a barrier -- the signatures encode which
		// side of the barrier-synthesis seam is allowed to touch tracked state, rather than
		// relying on `mutable` fields that every const-holding caller could otherwise corrupt.
		[[nodiscard]] std::shared_ptr<PhysicalTexture> GetTexture(ResourceId id) {
			auto it = m_textures.find(id);
			return it != m_textures.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<PhysicalTexture> GetTexture() {
			return GetTexture(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalBuffer> GetBuffer(ResourceId id) const {
			auto it = m_buffers.find(id);
			return it != m_buffers.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<const PhysicalBuffer> GetBuffer() const {
			return GetBuffer(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<PhysicalBuffer> GetBuffer(ResourceId id) {
			auto it = m_buffers.find(id);
			return it != m_buffers.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<PhysicalBuffer> GetBuffer() {
			return GetBuffer(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalAccelerationStructure>
		GetAccelerationStructure(ResourceId id) const {
			auto it = m_accelStructs.find(id);
			return it != m_accelStructs.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<const PhysicalAccelerationStructure> GetAccelerationStructure() const {
			return GetAccelerationStructure(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<PhysicalAccelerationStructure> GetAccelerationStructure(ResourceId id) {
			auto it = m_accelStructs.find(id);
			return it != m_accelStructs.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<PhysicalAccelerationStructure> GetAccelerationStructure() {
			return GetAccelerationStructure(IdOf<K>());
		}

		// Analysis & Provisioning: computes lifetimes for all realizations in the compiled
		// schedule, then provisions textures and buffers. Supports transient aliasing across
		// non-overlapping stage lifetimes when enabled (see ImageAliasPool/BufferAliasPool in
		// PhysicalResource.hpp).
		//
		// Lifetime is measured in stage indices, not flat node-slot indices: a resource touched
		// anywhere within stage N is "alive during stage N". That's coarser than per-node, but
		// it's the correct granularity given a Schedule -- nodes sharing a stage are provably
		// independent and may run concurrently, so a lifetime window can never straddle a single
		// stage boundary in a way that would let two live-at-once resources alias.
		void Provision(const Schedule& schedule, std::span<const Recipe> recipes, bool enableAliasing = true) {
			std::unordered_map<ResourceId, TemporalLifetime> lifetimes;
			for (std::size_t stageIndex = 0; stageIndex < schedule.stages.size(); ++stageIndex) {
				// Same-stage conflicting-layout detection: two nodes sharing a stage are provably
				// independent (ScheduleStage's documented invariant, Graph.hpp) and therefore
				// unschedulable relative to each other. If they derive different target layouts
				// for the same image key, that's a graph-authoring bug -- not a gap a barrier can
				// paper over -- so this throws rather than silently letting BarrierBatch::Add's
				// access-promotion (Execution.hpp) pick a winner. Scoped to images: a buffer has
				// no layout to conflict over.
				std::unordered_map<ResourceId, vk::ImageLayout> stageImageLayouts;

				for (std::size_t nodeIndex : schedule.stages[stageIndex].nodes) {
					if (nodeIndex >= recipes.size()) {
						continue;
					}
					const auto& recipe = recipes[nodeIndex];
					if (!recipe.isActive) {
						continue;
					}

					for (const auto& r : recipe.realizations) {
						auto& lt = lifetimes[r.key];
						lt.firstPass = std::min(lt.firstPass, stageIndex);
						lt.lastPass = std::max(lt.lastPass, stageIndex);

						// Same-stage layout conflicts only apply to images -- buffers have no
						// layout, and an AccelerationStructure realization has no ResourceDesc
						// fields DeriveImageState could meaningfully interpret at all (see
						// PhysicalAccelerationStructure's class comment). Skip anything that
						// isn't an image, rather than skip only Buffer, so a new non-image kind
						// added later doesn't silently fall through into DeriveImageState.
						if (r.desc.kind != ResourceDesc::Kind::Image2D && r.desc.kind != ResourceDesc::Kind::Image3D) {
							continue;
						}
						const vk::ImageLayout derivedLayout = DeriveImageState(
																  r.access,
																  recipe.domain,
																  r.desc.usageMask
																	  ? vk::ImageUsageFlags(r.desc.usageMask)
																	  : vk::ImageUsageFlags{},
																  static_cast<vk::Format>(r.desc.formatCode)
						)
																  .layout;

						auto [it, inserted] = stageImageLayouts.try_emplace(r.key, derivedLayout);
						if (!inserted && it->second != derivedLayout) {
							throw std::runtime_error(
								"PhysicalResourceRegistry::Provision: two nodes in the same schedule "
								"stage derive different target layouts for resource '" +
								std::string(r.key->name) + "'"
							);
						}
					}
				}
			}

			std::unordered_map<ResourceId, ResourceRealization> realizationsToProvision;
			for (const auto& stage : schedule.stages) {
				for (std::size_t nodeIndex : stage.nodes) {
					if (nodeIndex >= recipes.size()) {
						continue;
					}
					const auto& recipe = recipes[nodeIndex];
					if (!recipe.isActive) {
						continue;
					}

					for (const auto& r : recipe.realizations) {
						// AccelerationStructure is always pre-registered by the caller (via
						// RegisterImportedAccelerationStructure) before Setup/Compile run -- there
						// is nothing to provision, and neither ProvisionBuffer nor
						// ProvisionTexture below knows how to handle this kind. Skip it outright
						// rather than falling into the image branch by default.
						if (r.desc.kind == ResourceDesc::Kind::AccelerationStructure) {
							continue;
						}

						auto texIt = m_textures.find(r.key);
						if (texIt != m_textures.end() && texIt->second->IsImported()) {
							continue;
						}
						auto bufIt = m_buffers.find(r.key);
						if (bufIt != m_buffers.end() && bufIt->second->IsImported()) {
							continue;
						}

						realizationsToProvision[r.key] = r;
					}
				}
			}

			for (const auto& [id, r] : realizationsToProvision) {
				if (r.desc.kind == ResourceDesc::Kind::Buffer) {
					ProvisionBuffer(id, r.desc, lifetimes[id], enableAliasing);
				} else { // Image2D / Image3D
					ProvisionTexture(id, r.desc, lifetimes[id], enableAliasing);
				}
			}
		}

		// Drops the registry's own references. If nothing else still holds a shared_ptr to a
		// given resource, its RAII destructor runs and the GPU objects are freed now -- but if
		// something else does still hold one (a subgraph, a debug view, ...), that resource
		// correctly outlives this call. That's the real-sharing tradeoff versus the old
		// DestroyResources(), which destroyed unconditionally and synchronously.
		void Reset() {
			m_textures.clear();
			m_buffers.clear();
			m_accelStructs.clear();
			m_bindlessIndices.clear();
			m_imagePool.Reset();
			m_bufferPool.Reset();
			m_nextBindlessIndex = 0;
		}

	private:
		void ProvisionTexture(ResourceId id, const ResourceDesc& desc, TemporalLifetime lifetime, bool enableAliasing) {
			auto existingIt = m_textures.find(id);
			if (existingIt != m_textures.end() && !existingIt->second->IsImported()) {
				const auto& existingDesc = existingIt->second->GetDesc();
				// usageMask/mips/layers matter here, not just extent+format: a usage change
				// (e.g. a resource gaining eStorage) with the same size would otherwise silently
				// keep the stale texture, and every barrier decision downstream derives from
				// GetDesc().usageMask (ResourceState.hpp's Derive*State), so a stale mask would
				// then quietly mis-barrier every future access to this key.
				if (existingDesc.width == desc.width && existingDesc.height == desc.height &&
				    existingDesc.depth == desc.depth && existingDesc.formatCode == desc.formatCode &&
				    existingDesc.usageMask == desc.usageMask && existingDesc.mips == desc.mips &&
				    existingDesc.layers == desc.layers) {
					return; // already provisioned with a matching description
				}
			}

			std::shared_ptr<PhysicalTexture> tex = enableAliasing
				? m_imagePool.Acquire(desc, lifetime.firstPass, lifetime.lastPass)
				: std::make_shared<PhysicalTexture>(m_device, m_allocator, desc);

			// Assign a bindless index if this texture is sampled and a global descriptor set has
			// been provided -- nothing wires an actual bindless set up engine-wide yet, so this
			// stays inert bookkeeping until something does.
			const auto usage = desc.usageMask ? vk::ImageUsageFlags(desc.usageMask) : vk::ImageUsageFlagBits::eSampled;
			if ((usage & vk::ImageUsageFlagBits::eSampled) && m_globalDescriptorSet) {
				const std::uint32_t bindlessIndex = m_nextBindlessIndex++;
				m_bindlessIndices[id] = bindlessIndex;

				// Derived by the same function BarrierTranslator uses (ResourceState.hpp), so
				// this descriptor's layout can never diverge from whatever layout the barrier
				// seam actually puts the image in -- a Storage|Sampled image (e.g.
				// ComputeStorageImageDesc) correctly resolves to eGeneral here too, which a
				// hardcoded eShaderReadOnlyOptimal would have gotten wrong. Assumes
				// Graphics-domain sampling; nothing currently threads through which domain will
				// actually bind this descriptor.
				const vk::ImageLayout   bindlessLayout = DeriveImageState(
															 AccessKind::Read,
															 ExecutionDomain::Graphics,
															 usage,
															 static_cast<vk::Format>(desc.formatCode)
				)
															 .layout;
				vk::DescriptorImageInfo imageInfo{{}, tex->GetView(), bindlessLayout};
				vk::WriteDescriptorSet  write{
					m_globalDescriptorSet,
					m_globalBinding,
					bindlessIndex,
					1,
					vk::DescriptorType::eCombinedImageSampler,
					&imageInfo,
				};
				m_device.updateDescriptorSets(write, {});
			}

			m_textures[id] = std::move(tex);
		}

		void ProvisionBuffer(ResourceId id, const ResourceDesc& desc, TemporalLifetime lifetime, bool enableAliasing) {
			auto existingIt = m_buffers.find(id);
			if (existingIt != m_buffers.end() && !existingIt->second->IsImported()) {
				if (existingIt->second->GetDesc().byteSize == desc.byteSize) {
					return;
				}
			}

			m_buffers[id] = enableAliasing ? m_bufferPool.Acquire(desc, lifetime.firstPass, lifetime.lastPass)
										   : std::make_shared<PhysicalBuffer>(m_device, m_allocator, desc);
		}

		vk::Device      m_device{};
		VmaAllocator    m_allocator = nullptr;
		ImageAliasPool  m_imagePool;
		BufferAliasPool m_bufferPool;

		std::unordered_map<ResourceId, std::shared_ptr<PhysicalTexture>>               m_textures;
		std::unordered_map<ResourceId, std::shared_ptr<PhysicalBuffer>>                m_buffers;
		std::unordered_map<ResourceId, std::shared_ptr<PhysicalAccelerationStructure>> m_accelStructs;
		std::unordered_map<ResourceId, std::uint32_t>                                  m_bindlessIndices;

		vk::DescriptorSet m_globalDescriptorSet{};
		std::uint32_t     m_globalBinding = 0;
		std::uint32_t     m_nextBindlessIndex = 0;
	};

} // namespace brassica::graph
