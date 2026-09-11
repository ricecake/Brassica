#pragma once

#include <algorithm>
#include <array>
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
	class PhysicalResourceRegistry: public BindlessIndexSource {
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

		// The engine's one bindless descriptor set and which binding holds each array -- see
		// Engine::InitGlobalDescriptors for where these are created. Bindings, not a single
		// number: unlike the old single-binding placeholder this replaces, a texture may need a
		// slot in the sampled-2D array, the sampled-2D-array array, or the storage-image array
		// (or two of them at once -- see PhysicalTexture::GetSampledBindlessIndex's comment), and
		// an acceleration structure needs its own array entirely.
		struct BindlessBindings {
			vk::DescriptorSet       set{};
			vk::DescriptorSetLayout layout{}; // needed at pipeline-creation time, not just bind time
			std::uint32_t           sampledImage2DBinding = 0;
			std::uint32_t           sampledImage2DArrayBinding = 0;
			std::uint32_t           storageImageBinding = 0;
			std::uint32_t           accelerationStructureBinding = 0;
		};

		void SetGlobalDescriptorSet(const BindlessBindings& bindings) {
			m_bindless = bindings;
			EnsureFallbackTexture();
		}

		// The one real vk::DescriptorSet/vk::DescriptorSetLayout backing every ctx.Index<K>()
		// lookup -- what PhysicalExecutionBackend::Execute reads to populate
		// NodeContext::globalSet/globalSetLayout, so a node can bind it (or build a pipeline
		// layout referencing it) without needing a raw pointer/reference to Engine itself. Both
		// null before SetGlobalDescriptorSet has run.
		[[nodiscard]] vk::DescriptorSet GetBindlessDescriptorSet() const { return m_bindless.set; }

		[[nodiscard]] vk::DescriptorSetLayout GetBindlessDescriptorSetLayout() const { return m_bindless.layout; }

		// Maps any VersionedKey<K, N> id to the id of the one physical resource every version
		// shares -- derived purely from ResourceTypeInfo::versionBase (ResourceKey.hpp), so a node
		// author never registers this: writing Modify<Swapchain, 2> is enough for GetTexture,
		// Provision, and every barrier to agree it's the same image Modify<Swapchain, 1> touched.
		// Loops rather than a single hop only because a version's base is itself always unversioned
		// (VersionOf<K, 0> collapses to K -- ResourceKey.hpp) -- one hop always suffices today, but
		// looping costs nothing and doesn't depend on that staying true. Deliberately does not
		// follow historyTarget: History<K> is a genuinely different, double-buffered resource (see
		// the temporal-pair storage below), not an alias.
		[[nodiscard]] ResourceId ResolveId(ResourceId id) const {
			while (id->versionBase) {
				id = id->versionBase;
			}
			return id;
		}

		// initialLayout/hasDefinedContents describe what the caller already knows about this
		// import -- see PhysicalTexture's Imported constructor. Defaults are right for a
		// resource re-imported fresh every frame (e.g. the swapchain: about to be fully
		// overwritten, so eUndefined/false is correct); a long-lived import (e.g. the terrain
		// clipmap) must pass its real state or its first touch each frame would derive an
		// eUndefined transition and discard it.
		//
		// frameIndex is only used to time-delay reuse of whatever bindless index the *previous*
		// occupant of this id held (RetireBindlessIndices) -- irrelevant for a resource that's
		// never bindlessly sampled (the swapchain, today), but every import goes through the same
		// path uniformly rather than special-casing "this one doesn't need it."
		void RegisterImportedTexture(
			ResourceId          id,
			vk::Image           image,
			vk::ImageView       view,
			const ResourceDesc& desc,
			vk::ImageLayout     initialLayout = vk::ImageLayout::eUndefined,
			bool                hasDefinedContents = false,
			std::uint64_t       frameIndex = 0
		) {
			const ResourceId resolvedId = ResolveId(id);
			auto             existingIt = m_textures.find(resolvedId);
			if (existingIt != m_textures.end()) {
				RetireBindlessIndices(*existingIt->second, frameIndex);
			}

			auto tex = std::make_shared<PhysicalTexture>(image, view, desc, initialLayout, hasDefinedContents);
			AssignAndWriteBindlessIndices(*tex);
			m_textures[resolvedId] = std::move(tex);
		}

		void RegisterImportedBuffer(
			ResourceId          id,
			vk::Buffer          buffer,
			const ResourceDesc& desc,
			bool                hasDefinedContents = false
		) {
			m_buffers[ResolveId(id)] = std::make_shared<PhysicalBuffer>(buffer, desc, hasDefinedContents);
		}

		template <ResourceRef K>
		void RegisterImportedTexture(
			vk::Image           image,
			vk::ImageView       view,
			const ResourceDesc& desc,
			vk::ImageLayout     initialLayout = vk::ImageLayout::eUndefined,
			bool                hasDefinedContents = false,
			std::uint64_t       frameIndex = 0
		) {
			RegisterImportedTexture(IdOf<K>(), image, view, desc, initialLayout, hasDefinedContents, frameIndex);
		}

		template <ResourceRef K>
		void RegisterImportedBuffer(vk::Buffer buffer, const ResourceDesc& desc, bool hasDefinedContents = false) {
			RegisterImportedBuffer(IdOf<K>(), buffer, desc, hasDefinedContents);
		}

		// No desc, no initial-state parameters: unlike a texture/buffer import, there is nothing
		// caller-supplied to describe here (see PhysicalAccelerationStructure's class comment,
		// PhysicalResource.hpp) -- the handle itself is the entire state.
		//
		// Unlike ProvisionTexture, there is no desc-match early return here -- a rebuilt TLAS
		// always replaces the PhysicalAccelerationStructure object outright (TerrainPass builds a
		// brand new handle each time; see BuildOrUpdateAccelerationStructure). The bindless index
		// is carried forward across that replacement rather than retired+reallocated: it names
		// the *logical* resource (e.g. TerrainTLAS), which doesn't change identity just because
		// its physical handle did, and the descriptor is only rewritten when the raw handle
		// actually changed -- BuildOrUpdateAccelerationStructure already does its own synchronous
		// waitIdle() before this is ever called, so there is no in-flight command buffer that
		// could still be reading the old handle through this slot.
		void RegisterImportedAccelerationStructure(ResourceId id, vk::AccelerationStructureKHR as) {
			const ResourceId    resolvedId = ResolveId(id);
			auto                existingIt = m_accelStructs.find(resolvedId);
			const bool          handleChanged = existingIt == m_accelStructs.end() || existingIt->second->Get() != as;
			const std::uint32_t index = existingIt != m_accelStructs.end() ? existingIt->second->GetBindlessIndex()
																		   : m_accelStructArena.Allocate();

			auto newAS = std::make_shared<PhysicalAccelerationStructure>(as);
			newAS->SetBindlessIndex(index);
			m_accelStructs[resolvedId] = newAS;

			if (handleChanged) {
				WriteAccelerationStructureDescriptor(index, as);
			}
		}

		template <ResourceRef K>
		void RegisterImportedAccelerationStructure(vk::AccelerationStructureKHR as) {
			RegisterImportedAccelerationStructure(IdOf<K>(), as);
		}

		// The sampled-array index for id -- the common case, matching what a shader's texture2D[]/
		// texture2DArray[] read uses. 0 (the permanent fallback texture, never assigned to a real
		// resource) covers both "id is not a texture" and "id's texture has no eSampled usage",
		// same as a texture that was simply never registered -- a caller doesn't need to
		// distinguish those, since either way there is nothing meaningful to sample.
		[[nodiscard]] std::uint32_t GetBindlessIndex(ResourceId id) const {
			auto tex = GetTexture(id);
			return tex ? tex->GetSampledBindlessIndex() : 0;
		}

		// BindlessIndexSource: the lookup NodeContext::Index<K>() (Execution.hpp) calls through
		// to. Checks the AS registry first since an AS's ResourceId never resolves via
		// GetTexture -- the two kinds are disjoint, so this is never ambiguous -- then falls
		// back to the sampled-image index, same as GetBindlessIndex(id) above. Deliberately
		// does not consider the storage-image arena -- StorageIndexOf below is the explicit way
		// a caller that already knows a resource is storage-only asks for that index instead,
		// rather than this one guessing which arena was meant.
		[[nodiscard]] std::uint32_t IndexOf(ResourceId id) const override {
			if (auto as = GetAccelerationStructure(id)) {
				return as->GetBindlessIndex();
			}
			return GetBindlessIndex(id);
		}

		// BindlessIndexSource: the lookup NodeContext::StorageIndex<K>() calls through to.
		// Unlike IndexOf, there's no AS/sampled fallback to check first -- a storage-image read
		// is never expected to resolve to anything else, so this is just GetStorageBindlessIndex
		// under the interface every node's Execute actually has access to (a NodeContext, not a
		// raw registry pointer).
		[[nodiscard]] std::uint32_t StorageIndexOf(ResourceId id) const override { return GetStorageBindlessIndex(id); }

		template <ResourceRef K>
		[[nodiscard]] std::uint32_t GetBindlessIndex() const {
			return GetBindlessIndex(IdOf<K>());
		}

		// The storage-image-array index for id. Only ever meaningful for a key a node already
		// knows is a storage image (e.g. a compute LUT it writes via imageStore). Index 0 is
		// reserved-but-unwritten here too (see BindlessArena) purely so "0" reliably means "never
		// assigned" for retirement bookkeeping -- unlike the sampled array, nothing ever writes a
		// real descriptor at slot 0 here, since a storage-image read is never expected to fall
		// back to anything.
		[[nodiscard]] std::uint32_t GetStorageBindlessIndex(ResourceId id) const {
			auto tex = GetTexture(id);
			return tex ? tex->GetStorageBindlessIndex() : 0;
		}

		template <ResourceRef K>
		[[nodiscard]] std::uint32_t GetStorageBindlessIndex() const {
			return GetStorageBindlessIndex(IdOf<K>());
		}

		// The acceleration-structure-array index for id.
		[[nodiscard]] std::uint32_t GetAccelerationStructureBindlessIndex(ResourceId id) const {
			auto as = GetAccelerationStructure(id);
			return as ? as->GetBindlessIndex() : 0;
		}

		template <ResourceRef K>
		[[nodiscard]] std::uint32_t GetAccelerationStructureBindlessIndex() const {
			return GetAccelerationStructureBindlessIndex(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalTexture> GetTexture(ResourceId id) const {
			auto it = m_textures.find(ResolveId(id));
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
			auto it = m_textures.find(ResolveId(id));
			return it != m_textures.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<PhysicalTexture> GetTexture() {
			return GetTexture(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalBuffer> GetBuffer(ResourceId id) const {
			auto it = m_buffers.find(ResolveId(id));
			return it != m_buffers.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<const PhysicalBuffer> GetBuffer() const {
			return GetBuffer(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<PhysicalBuffer> GetBuffer(ResourceId id) {
			auto it = m_buffers.find(ResolveId(id));
			return it != m_buffers.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<PhysicalBuffer> GetBuffer() {
			return GetBuffer(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<const PhysicalAccelerationStructure>
		GetAccelerationStructure(ResourceId id) const {
			auto it = m_accelStructs.find(ResolveId(id));
			return it != m_accelStructs.end() ? it->second : nullptr;
		}

		template <ResourceRef K>
		[[nodiscard]] std::shared_ptr<const PhysicalAccelerationStructure> GetAccelerationStructure() const {
			return GetAccelerationStructure(IdOf<K>());
		}

		[[nodiscard]] std::shared_ptr<PhysicalAccelerationStructure> GetAccelerationStructure(ResourceId id) {
			auto it = m_accelStructs.find(ResolveId(id));
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
		//
		// frameIndex selects which of a History<K> pair's two physical slots K and History<K>
		// resolve to this frame -- see ProvisionTemporalPairs below.
		void Provision(
			const Schedule&         schedule,
			std::span<const Recipe> recipes,
			std::uint64_t           frameIndex,
			bool                    enableAliasing = true
		) {
			ProvisionTemporalPairs(recipes, frameIndex);

			// Bindless indices retired by an earlier Provision() (a texture replaced by a resize,
			// most likely -- see ProvisionTexture) become reusable once enough frames have passed
			// that nothing in flight can still reference them. See BindlessArena's comment.
			m_sampledArena.ProcessRetirements(frameIndex);
			m_sampledArrayArena.ProcessRetirements(frameIndex);
			m_storageArena.ProcessRetirements(frameIndex);
			m_accelStructArena.ProcessRetirements(frameIndex);

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
						// Resolved once and used throughout: every version of a key shares one
						// lifetime, one same-stage layout check, and one physical resource -- see
						// ResolveId's comment above.
						const ResourceId resolvedKey = ResolveId(r.key);

						auto& lt = lifetimes[resolvedKey];
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

						auto [it, inserted] = stageImageLayouts.try_emplace(resolvedKey, derivedLayout);
						if (!inserted && it->second != derivedLayout) {
							throw std::runtime_error(
								"PhysicalResourceRegistry::Provision: two nodes in the same schedule "
								"stage derive different target layouts for resource '" +
								std::string(resolvedKey->name) + "'"
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

						const ResourceId resolvedKey = ResolveId(r.key);

						auto texIt = m_textures.find(resolvedKey);
						if (texIt != m_textures.end() && texIt->second->IsImported()) {
							continue;
						}
						auto bufIt = m_buffers.find(resolvedKey);
						if (bufIt != m_buffers.end() && bufIt->second->IsImported()) {
							continue;
						}

						realizationsToProvision[resolvedKey] = r;
					}
				}
			}

			for (const auto& [id, r] : realizationsToProvision) {
				if (r.desc.kind == ResourceDesc::Kind::Buffer) {
					ProvisionBuffer(id, r.desc, lifetimes[id], enableAliasing);
				} else { // Image2D / Image3D
					ProvisionTexture(id, r.desc, lifetimes[id], enableAliasing, frameIndex);
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
			m_temporalTexturePairs.clear();
			m_temporalBufferPairs.clear();
			m_imagePool.Reset();
			m_bufferPool.Reset();
			m_sampledArena.Reset();
			m_sampledArrayArena.Reset();
			m_storageArena.Reset();
			m_accelStructArena.Reset();
			m_fallbackTexture.reset();
		}

	private:
		// A monotonically-growing bindless-array index space with recycling. Index 0 is never
		// handed out by Allocate() -- reserved universally (not just in the sampled array, which
		// actually writes a real fallback texture there) purely so "index == 0" reliably means
		// "never assigned" wherever a caller needs to ask that, e.g. RetireBindlessIndices below.
		//
		// Retire() doesn't free immediately: a resource that's just been replaced (a window
		// resize changing every extent-derived desc, most likely) might still be referenced by a
		// command buffer recorded before the replacement and not yet retired by the GPU, so a
		// released index is held for kRetireDelayFrames before Allocate() can hand it out again.
		// Belt-and-braces -- today's one path that changes descs (Engine::RecreateSwapchain)
		// already calls device.waitIdle() before anything reaches Provision() again, so nothing
		// currently exercises the delay for real -- but the invariant belongs here, not in
		// whichever caller happens to be idle-safe today.
		class BindlessArena {
		public:
			[[nodiscard]] std::uint32_t Allocate() {
				if (!m_freeList.empty()) {
					std::uint32_t index = m_freeList.back();
					m_freeList.pop_back();
					return index;
				}
				return m_nextIndex++;
			}

			void Retire(std::uint32_t index, std::uint64_t frameIndex) {
				m_retiring.push_back({frameIndex + kRetireDelayFrames, index});
			}

			void ProcessRetirements(std::uint64_t frameIndex) {
				std::erase_if(m_retiring, [&](const auto& entry) {
					if (entry.first > frameIndex) {
						return false;
					}
					m_freeList.push_back(entry.second);
					return true;
				});
			}

			void Reset() {
				m_nextIndex = 1;
				m_freeList.clear();
				m_retiring.clear();
			}

		private:
			static constexpr std::uint64_t kRetireDelayFrames = 2;

			std::uint32_t                                        m_nextIndex{1};
			std::vector<std::uint32_t>                           m_freeList;
			std::vector<std::pair<std::uint64_t, std::uint32_t>> m_retiring;
		};

		void
		WriteSampledImageDescriptor(std::uint32_t index, vk::ImageView view, vk::ImageLayout layout, bool isArray) {
			if (!m_bindless.set) {
				return;
			}
			vk::DescriptorImageInfo imageInfo{{}, view, layout};
			vk::WriteDescriptorSet  write{
				m_bindless.set,
				isArray ? m_bindless.sampledImage2DArrayBinding : m_bindless.sampledImage2DBinding,
				index,
				1,
				vk::DescriptorType::eSampledImage,
				&imageInfo,
			};
			m_device.updateDescriptorSets(write, {});
		}

		void WriteStorageImageDescriptor(std::uint32_t index, vk::ImageView view) {
			if (!m_bindless.set) {
				return;
			}
			// Always eGeneral: the only layout a storage image is ever legally accessed through
			// (DeriveImageState's Storage-usage branches, ResourceState.hpp, agree).
			vk::DescriptorImageInfo imageInfo{{}, view, vk::ImageLayout::eGeneral};
			vk::WriteDescriptorSet  write{
				m_bindless.set,
				m_bindless.storageImageBinding,
				index,
				1,
				vk::DescriptorType::eStorageImage,
				&imageInfo,
			};
			m_device.updateDescriptorSets(write, {});
		}

		void WriteAccelerationStructureDescriptor(std::uint32_t index, vk::AccelerationStructureKHR as) {
			if (!m_bindless.set) {
				return;
			}
			vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
			asInfo.setAccelerationStructures(as);
			vk::WriteDescriptorSet write{
				m_bindless.set,
				m_bindless.accelerationStructureBinding,
				index,
				1,
				vk::DescriptorType::eAccelerationStructureKHR,
				nullptr,
			};
			write.pNext = &asInfo;
			m_device.updateDescriptorSets(write, {});
		}

		// Assigns and writes whichever of the sampled/storage arrays this texture's usage calls
		// for -- both, if it has both (see PhysicalTexture::GetSampledBindlessIndex's comment).
		// No-ops entirely if SetGlobalDescriptorSet was never called, so a registry used the way
		// every test before this one used it (no bindless set at all) behaves exactly as before.
		void AssignAndWriteBindlessIndices(PhysicalTexture& tex) {
			if (!m_bindless.set) {
				return;
			}
			const auto& desc = tex.GetDesc();
			const auto  usage = desc.usageMask ? vk::ImageUsageFlags(desc.usageMask) : vk::ImageUsageFlags{};
			const bool  isArray = desc.layers > 1;

			if (usage & vk::ImageUsageFlagBits::eSampled) {
				// A plain-2D and a 2D-array texture write into two independently-sized Vulkan
				// arrays (sampledImage2DBinding vs sampledImage2DArrayBinding) -- sharing one
				// counter between them would let a 2D-array index grow past that binding's own
				// (typically much smaller) descriptorCount and spill into whatever binding
				// happens to sit next, which the validation layer reports as a descriptorType
				// mismatch on the overflowed write, not as an out-of-range index.
				BindlessArena&      arena = isArray ? m_sampledArrayArena : m_sampledArena;
				const std::uint32_t index = arena.Allocate();
				tex.SetSampledBindlessIndex(index);
				// Derived by the same function BarrierTranslator uses (ResourceState.hpp), so this
				// descriptor's layout can never diverge from whatever layout the barrier seam
				// actually puts the image in -- a Storage|Sampled image (e.g.
				// ComputeStorageImageDesc) correctly resolves to eGeneral here too, matching its
				// storage-array entry, which a hardcoded eShaderReadOnlyOptimal would have gotten
				// wrong. Assumes Graphics-domain sampling; nothing currently threads through which
				// domain will actually bind this descriptor.
				const vk::ImageLayout layout = DeriveImageState(
												   AccessKind::Read,
												   ExecutionDomain::Graphics,
												   usage,
												   static_cast<vk::Format>(desc.formatCode)
				)
												   .layout;
				WriteSampledImageDescriptor(index, tex.GetView(), layout, isArray);
			}
			if (usage & vk::ImageUsageFlagBits::eStorage) {
				const std::uint32_t index = m_storageArena.Allocate();
				tex.SetStorageBindlessIndex(index);
				WriteStorageImageDescriptor(index, tex.GetView());
			}
		}

		// Called just before a texture is replaced outright (ProvisionTexture's desc-mismatch
		// path, or RegisterImportedTexture overwriting a prior import) -- its old slots become
		// reusable once ProcessRetirements confirms enough frames have passed, not immediately.
		void RetireBindlessIndices(const PhysicalTexture& tex, std::uint64_t frameIndex) {
			if (tex.GetSampledBindlessIndex() != 0) {
				// Must route to the same arena AssignAndWriteBindlessIndices originally drew from
				// -- recomputed from the desc rather than cached anywhere, since it's already the
				// single source of truth for which array a texture's sampled index lives in.
				BindlessArena& arena = tex.GetDesc().layers > 1 ? m_sampledArrayArena : m_sampledArena;
				arena.Retire(tex.GetSampledBindlessIndex(), frameIndex);
			}
			if (tex.GetStorageBindlessIndex() != 0) {
				m_storageArena.Retire(tex.GetStorageBindlessIndex(), frameIndex);
			}
		}

		// A permanent 1x1 texture at sampled-array index 0, so nonuniformEXT-indexing a key that
		// was never registered (GetBindlessIndex's 0-on-miss) reads a real, harmless descriptor
		// instead of hitting undefined behavior. Created lazily, on the first
		// SetGlobalDescriptorSet call that actually provides a set -- its contents/layout are
		// left for a future stage to actually initialize (nothing samples the bindless array yet
		// at this point in the migration); only its descriptor identity is real from here on.
		void EnsureFallbackTexture() {
			if (m_fallbackTexture) {
				return;
			}
			ResourceDesc desc{
				.kind = ResourceDesc::Kind::Image2D,
				.width = 1,
				.height = 1,
				.formatCode = static_cast<std::uint32_t>(vk::Format::eR8G8B8A8Unorm),
				.usageMask = static_cast<std::uint32_t>(
					vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst
				),
			};
			m_fallbackTexture = std::make_shared<PhysicalTexture>(m_device, m_allocator, desc);
			const std::uint32_t index = m_sampledArena.Allocate();
			m_fallbackTexture->SetSampledBindlessIndex(index);
			WriteSampledImageDescriptor(
				index,
				m_fallbackTexture->GetView(),
				vk::ImageLayout::eShaderReadOnlyOptimal,
				false
			);
		}

		// A History<K> realization means "last frame's K" -- a genuinely different, persistent
		// image than this frame's K, not an alias (ResolveId deliberately never follows
		// historyTarget). Ping-ponged: two real physical resources, K resolving to one and
		// History<K> to the other, swapping which is which every frame. Bypasses
		// ProvisionTexture/ProvisionBuffer's alias pool unconditionally, regardless of
		// enableAliasing -- pool memory is reassignable across frames (see the alias-pool lifetime
		// caveat this same header's Provision loop does not protect against), which would destroy
		// exactly the contents History exists to preserve.
		//
		// Runs once per Provision() call, before the ordinary collection/provisioning loops below:
		// by the time those loops see K or History<K>, m_textures/m_buffers already holds this
		// frame's correctly-assigned slot for each, and ProvisionTexture/ProvisionBuffer's own
		// desc-match early return (see below) leaves it untouched rather than reprovisioning it.
		void ProvisionTemporalPairs(std::span<const Recipe> recipes, std::uint64_t frameIndex) {
			std::unordered_map<ResourceId, ResourceId>   historyIdForBase;
			std::unordered_map<ResourceId, ResourceDesc> descForBase;
			for (const auto& recipe : recipes) {
				for (const auto& r : recipe.realizations) {
					if (!r.key->isHistory) {
						continue;
					}
					const ResourceId base = r.key->historyTarget;
					historyIdForBase[base] = r.key;
					descForBase[base] = r.desc; // PreviousFrame has no realizations of its own --
					                            // the reader's desc is the only one available.
				}
			}

			const std::size_t parity = static_cast<std::size_t>(frameIndex % 2);
			for (const auto& [base, historyId] : historyIdForBase) {
				const ResourceDesc& desc = descForBase.at(base);
				if (desc.kind == ResourceDesc::Kind::Buffer) {
					auto& pair = m_temporalBufferPairs[base];
					if (!pair[0]) {
						pair[0] = std::make_shared<PhysicalBuffer>(m_device, m_allocator, desc);
						pair[1] = std::make_shared<PhysicalBuffer>(m_device, m_allocator, desc);
					}
					m_buffers[base] = pair[parity];
					m_buffers[historyId] = pair[1 - parity];
				} else {
					auto& pair = m_temporalTexturePairs[base];
					if (!pair[0]) {
						pair[0] = std::make_shared<PhysicalTexture>(m_device, m_allocator, desc);
						pair[1] = std::make_shared<PhysicalTexture>(m_device, m_allocator, desc);
						// Bypasses ProvisionTexture entirely (this pair's own !pair[0] check is
						// its equivalent of ProvisionTexture's desc-match early return), so it
						// must assign bindless indices itself -- otherwise a History<K> pair
						// would silently never get one at all.
						AssignAndWriteBindlessIndices(*pair[0]);
						AssignAndWriteBindlessIndices(*pair[1]);
					}
					m_textures[base] = pair[parity];
					m_textures[historyId] = pair[1 - parity];
				}
			}
		}

		void ProvisionTexture(
			ResourceId          id,
			const ResourceDesc& desc,
			TemporalLifetime    lifetime,
			bool                enableAliasing,
			std::uint64_t       frameIndex
		) {
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
				// About to replace this texture (e.g. a resize) -- its bindless slots, if any,
				// become reusable once ProcessRetirements confirms it's safe, not immediately.
				RetireBindlessIndices(*existingIt->second, frameIndex);
			}

			std::shared_ptr<PhysicalTexture> tex = enableAliasing
				? m_imagePool.Acquire(desc, lifetime.firstPass, lifetime.lastPass)
				: std::make_shared<PhysicalTexture>(m_device, m_allocator, desc);

			AssignAndWriteBindlessIndices(*tex);

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

		// Keyed by the History target's base id (never by the History<K> id itself). Both slots
		// are Owning and allocated once, lazily, on the first frame that declares a History<K>
		// realization for this base -- see ProvisionTemporalPairs.
		std::unordered_map<ResourceId, std::array<std::shared_ptr<PhysicalTexture>, 2>> m_temporalTexturePairs;
		std::unordered_map<ResourceId, std::array<std::shared_ptr<PhysicalBuffer>, 2>>  m_temporalBufferPairs;

		BindlessBindings m_bindless{};
		BindlessArena    m_sampledArena;      // sampledImage2DBinding
		BindlessArena    m_sampledArrayArena; // sampledImage2DArrayBinding -- see AssignAndWriteBindlessIndices
		BindlessArena    m_storageArena;
		BindlessArena    m_accelStructArena;
		std::shared_ptr<PhysicalTexture> m_fallbackTexture;
	};

} // namespace brassica::graph
