#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph/ResourceKey.hpp"

#ifdef MemoryBarrier
	#undef MemoryBarrier
#endif

// This is the entire Vulkan seam for the graph skeleton. Nothing in this file, or anything
// it includes, may depend on vulkan/vulkan.hpp, VMA, or external/FrameGraph. When a real
// backend lands, this is the only header whose stubs get replaced -- everything else in
// include/graph/ only ever sees these types.

namespace brassica::graph {

	enum class ExecutionDomain : std::uint8_t { Graphics, Compute, Transfer, Host };

	// Bit for a GPU queue's slot in a resource's queue-touch mask (PhysicalTexture/Buffer/
	// AccelerationStructure's m_queueTouchMask, PhysicalResource.hpp) -- Graphics/Compute/
	// Transfer only, relying on their declaration order above as bit index. Host never
	// participates: it has no GPU queue and never records a barrier. Used both to update a
	// resource's mask after a touch and to test the mask against the queue about to record one
	// (BarrierTranslator.hpp's cross-queue sanitization).
	inline constexpr std::uint8_t QueueBit(ExecutionDomain domain) {
		return static_cast<std::uint8_t>(1u << static_cast<unsigned>(domain));
	}

	// Every GPU queue bit at once -- the conservative "assume the worst" default a resource's
	// queue-touch mask starts from (PhysicalResource.hpp), matching the existing "assume
	// eAllCommands already touched it" reasoning the Aliased-memory-reused and Imported-with-
	// defined-contents constructors already use for lastStage/lastAccess.
	inline constexpr std::uint8_t kAllQueuesTouchMask =
		QueueBit(ExecutionDomain::Graphics) | QueueBit(ExecutionDomain::Compute) | QueueBit(ExecutionDomain::Transfer);

	inline constexpr std::uint32_t kQueueFamilyIgnored = 0xFFFFFFFFu;

	struct QueueInfo {
		void*         queue = nullptr; // Opaque VkQueue
		std::uint32_t familyIndex = kQueueFamilyIgnored;
	};

	struct QueueSet {
		QueueInfo graphics{};
		QueueInfo compute{};
		QueueInfo transfer{};
	};

	enum class AccessKind : std::uint8_t { Read, Write, ReadWrite };

	// Governs whether a resource's underlying buffer is CPU-writable, and if so, how a write
	// reaches the GPU. None (the default): today's behavior exactly, device-local only, no
	// host-mapping -- zero change for any existing caller. Staged: the buffer/texture stays
	// device-local; a CPU write goes through a registry-owned staging buffer and a real
	// copyBuffer/copyBufferToImage on the graph's own command buffer -- zero consumer-side
	// obligations, the destination handle never changes. Mapped (buffers only): one
	// PhysicalBuffer with FRAME_OVERLAP CPU-visible sub-ranges and no copy command at all -- an
	// explicit opt-in for a node author willing to index data[frameIndex % FRAME_OVERLAP]
	// themselves in-shader, in exchange for skipping the staging copy.
	enum class HostAccess : std::uint8_t { None, Staged, Mapped };

	// Coarse ordering between nodes, independent of resource flow: "this pass belongs after
	// deferred shading" without naming everything deferred shading touches. Int-backed and spaced
	// so callers can insert a new band between existing ones without renumbering anything, the same
	// way ExecutionDomain leaves room for future domains. Comparison is always on the underlying
	// int; the two extremes exist so PreviousFrame/NextFrame (Frame.hpp) bracket every other node
	// with no risk of a user-defined phase colliding with them.
	//
	// This layer only defines the generic bands. An engine's own named phases (e.g.
	// DeferredShading, ForwardTranslucent) are its vocabulary, not the graph's -- see
	// include/passes/RenderPhases.hpp.
	enum class Phase : std::int32_t {
		PreviousFrame = INT32_MIN,
		Early = -1000,
		Default = 0,
		Late = 1000,
		NextFrame = INT32_MAX,
	};

	// Stand-in for a backend command buffer. Deliberately opaque at this layer.
	struct CommandBuffer {
		void* vkCmd = nullptr;
	};

	// The concrete realization of a resource key for this frame, factoring in things like
	// screen resolution -- the "more concrete type" a node produces at Setup time, as opposed
	// to the abstract key it declares at the class level. formatCode/usageMask are opaque
	// here on purpose: this layer never interprets them, it only carries them through (and,
	// later, compares/hashes them for aliasing) for a backend to map onto real enums.
	struct ResourceDesc {
		// AccelerationStructure carries none of the fields below -- no extent/format/usage is
		// meaningful for it, they stay at their defaults. See PhysicalResource.hpp's
		// PhysicalAccelerationStructure comment for why this kind is Imported-only.
		enum class Kind : std::uint8_t { Image2D, Image3D, Buffer, AccelerationStructure };

		Kind          kind = Kind::Image2D;
		std::uint32_t width = 0;
		std::uint32_t height = 0;
		std::uint32_t depth = 1;
		std::uint32_t mips = 1;
		std::uint32_t layers = 1;
		std::uint32_t formatCode = 0;
		std::uint32_t usageMask = 0;
		std::uint64_t byteSize = 0;
		HostAccess    hostAccess = HostAccess::None;

		// True for a resource meant to live indefinitely once allocated, surviving frames where
		// the producing node is inactive (e.g. TerrainBiomeNode's climate state, regenerated only
		// every Nth frame). PhysicalResourceRegistry::ProvisionTexture/ProvisionBuffer never route
		// a persistent realization through the alias pool, regardless of enableAliasing -- a pool
		// block's "last live stage" is only ever updated when Acquire() is called again for that
		// id, which never happens once a desc-match keeps reusing the same handle across frames
		// (ProvisionTexture/Buffer's early return). Left unmarked, the pool would eventually
		// believe the block is free and hand that exact memory to an unrelated resource while this
		// one is still alive, silently corrupting it. A node declares this by setting the flag on
		// the ResourceDesc its Setup() builds -- in practice, baked into that resource's shared
		// desc-builder function so every Create/Modify of the key gets it for free.
		bool persistent = false;
	};

	// Moved up from just below Recipe (where it's still used) so NodeContext's undeclared-access
	// guard (CheckDeclared, below) can hold a std::span<const ResourceRealization> over a node's
	// own Recipe::realizations -- NodeContext is defined before Recipe, so this type needs to be
	// complete by the time NodeContext's inline method bodies reference it.
	struct ResourceRealization {
		ResourceId   key = nullptr;
		AccessKind   access = AccessKind::Read;
		ResourceDesc desc{};

		// Used only when this realization becomes a color attachment with loadOp=eClear (see
		// DynamicRenderingWrapper::Begin, PhysicalExecutionBackend.hpp). Defaults to opaque
		// black, matching what every current color target except the G-buffer wants. The
		// G-buffer's alpha channel is a data sentinel, not just opacity -- deferred.frag reads
		// albedo.a < 0.01 to mean "no terrain rendered here, show the gradient background" -- so
		// TerrainNode overrides this to {0,0,0,0} for its three color realizations. Getting this
		// wrong doesn't fail to compile or throw; it just silently miscomposites, which is
		// exactly what happened before this field existed (see the migration's post-port fix).
		std::array<float, 4> clearColor{0.0f, 0.0f, 0.0f, 1.0f};

		// Used only when this realization's AttachmentRole (ResourceState.hpp) is ShadingRate --
		// VkRenderingFragmentShadingRateAttachmentInfoKHR::shadingRateAttachmentTexelSize, which
		// has no principled default this layer could pick on its own (it's a per-device minimum,
		// see ShadingRateAttachmentDesc's comment, PhysicalResource.hpp). 16x16 matches the most
		// common reported minimum; a node targeting a device with a different minimum must set
		// this explicitly.
		std::array<std::uint32_t, 2> shadingRateTexelSize{16, 16};

		// Set via AccessHandle::AnyOrder() -- the author is asserting that a conflicting write
		// against this exact key from another AnyOrder() access needs no relation, because the
		// two commute (e.g. depth-tested opaque geometry writing disjoint screen regions). Without
		// this, two writers of one key with nothing ordering them is an ambiguity error (design
		// plan's §E); this is the single opt-out, and it only suppresses that specific check -- a
		// real data race from an actual conflicting write still corrupts output the same as ever,
		// this is not a correctness guarantee, just a declaration that the author already checked.
		bool anyOrder = false;

		// Set via AccessHandle::Required() on a Read access -- design plan's §E: this node is
		// inactive unless an active writer W of this key satisfies W <= this node in the schedule
		// order (a writer that only runs *after* this node doesn't count; that would make
		// activity depend on something that hasn't happened yet). Meaningless on a Write/
		// ReadWrite access -- a node always "has" whatever it itself just wrote. Distinct from
		// Requires<A>(), which couples activity to a specific *node*; this couples it to whether
		// a specific *key* actually has live data flowing into this read.
		bool required = false;
	};

	// Opaque per-resource bindless-index lookup *and* CPU-write entry point. Implemented by
	// PhysicalResourceRegistry (PhysicalRegistry.hpp, Vulkan-aware) so that
	// NodeContext::Index<K>()/StorageIndex<K>()/Write<K>()/WriteSpan<K>() below can live in this
	// Vulkan-free seam without this file knowing PhysicalResourceRegistry, or Vulkan, exists --
	// the same inversion Resource/MemoryBarrier already use for barriers. Named ResourceServices,
	// not BindlessIndexSource (its name until this stage): WaitIdle already meant this interface
	// was more than bindless-index lookups before BeginHostWrite/EndHostWrite below made that
	// official. UploadPredefinedBuffer/UploadPredefinedTexture used to live here too -- removed
	// once HostWriteNode<Key,T> (Node.hpp) and Write<K>/WriteSpan<K> gave every real caller a
	// path onto the graph's own per-frame command buffer instead; the one caller that still
	// needs to upload outside any frame (Util.hpp's CreateAndRegisterStaticBuffer/Texture, a
	// pre-frame-loop static-asset path) calls PhysicalResourceRegistry's own
	// UploadBufferImmediate/UploadTextureImmediate directly -- not through this interface, since
	// nothing else needs them.
	//
	// Two lookups, not one: a texture with both eSampled and eStorage usage (a compute LUT
	// written via imageStore and later sampled by a different node, e.g. the atmosphere LUTs)
	// occupies a slot in both the sampled and storage bindless arenas at once -- IndexOf and
	// StorageIndexOf are how a caller says which one it means, since one flat index can't name
	// both (see PhysicalTexture::GetSampledBindlessIndex's comment, PhysicalResource.hpp).
	struct ResourceServices {
		virtual ~ResourceServices() = default;
		virtual std::uint32_t IndexOf(ResourceId) const = 0;

		// mip defaults to 0, the only value that was ever valid before per-mip storage views
		// existed -- a resource with desc.mips == 1 (everything except the terrain min/max
		// chain) ignores it entirely. For a resource with real per-mip storage usage, this is
		// which single-mip view's bindless slot to return.
		virtual std::uint32_t StorageIndexOf(ResourceId, std::uint32_t mip = 0) const = 0;

		// The bindless-storage-buffer analogue of StorageIndexOf, for a buffer registered through
		// PhysicalResourceRegistry::ProvisionBuffer with eStorageBuffer usage (PhysicalRegistry.hpp's
		// AssignAndWriteBindlessBufferIndex). Non-pure, defaulted to 0 ("never assigned") unlike
		// IndexOf/StorageIndexOf: this capability was added after several other ResourceServices
		// implementations already existed (test fixtures, mainly), and none of them need to grow a
		// buffer-bindless arena just to keep compiling.
		virtual std::uint32_t BufferIndexOf(ResourceId) const { return 0u; }

		virtual void WaitIdle() {}

		// Begins a CPU write into `id`'s current-frame slot -- valid only for a resource whose
		// ResourceDesc::hostAccess is Staged or Mapped (HostAccess::None, the default, returns an
		// empty span; there is nothing to write into). sizeBytes is the logical size the caller
		// intends to write, must not exceed the resource's real per-slot capacity.
		// insideRendering must be true exactly when called from within a dynamic-rendering scope
		// (DynamicRenderingWrapper::Begin/End) -- a Staged write records a real copy command in
		// EndHostWrite, which vkCmdCopyBuffer/vkCmdCopyBufferToImage can't do there; a Mapped
		// write never touches the command buffer at all, so it's unaffected either way.
		// domain is the calling node's own ExecutionDomain -- a Staged write's real copy command
		// always executes at the transfer stage (EndHostWrite's vkCmdCopyBuffer/CopyBufferToImage
		// below), but the Acquire pre-barrier that runs before Execute() derives the stage/access
		// it leaves in tracked state from this domain, not from "there happens to be a transfer
		// copy inside." A Transfer-domain node's pre-barrier already lands on eAllTransfer/
		// eTransferWrite, which matches; any other domain's pre-barrier lands on that domain's own
		// stage (e.g. eColorAttachmentOutput for Graphics), which doesn't -- the next consumer's
		// srcStage would then miss the real transfer write. PhysicalResourceRegistry's override
		// throws on that combination rather than letting it corrupt tracked state. Returns an
		// empty span by default (matching every other virtual here, all no-ops until
		// PhysicalResourceRegistry overrides them).
		virtual std::span<std::byte> BeginHostWrite(
			ResourceId /*id*/,
			std::size_t /*sizeBytes*/,
			std::uint64_t /*frameIndex*/,
			bool /*insideRendering*/,
			ExecutionDomain /*domain*/
		) {
			return {};
		}

		// Finalizes the write BeginHostWrite began for the same id this frame: for a Staged
		// resource, records the real copy into cmd (the graph's own command buffer for this
		// frame); for a Mapped resource, flushes the written slice in case the underlying memory
		// isn't coherent. No-op default.
		virtual void EndHostWrite(ResourceId, CommandBuffer) {}
	};

	// Per-node execution state, passed to a node's Execute in place of a bare CommandBuffer.
	// pipeline/pipelineLayout/frameSet/frameSetLayout/globalSet/globalSetLayout are opaque
	// Vulkan handles (same trick as CommandBuffer::vkCmd) -- real bind/push/draw calls are free
	// functions in the Vulkan-aware include/render/NodeCommands.hpp, not members here.
	struct NodeContext {
		CommandBuffer cmd{};
		std::uint32_t width = 0;
		std::uint32_t height = 0;
		std::uint64_t frameIndex = 0;
		void*         pipeline = nullptr;
		void*         pipelineLayout = nullptr;

		// Set 0, always bound: just the per-frame UBO (camera/time/frame data). Genuinely
		// double-buffered on the physical side (one instance per frame-in-flight) since its
		// contents are CPU-written fresh every frame -- see PhysicalResourceRegistry's comment
		// on why this is deliberately a *different* descriptor set than the bindless one below,
		// not folded into it.
		void* frameSet = nullptr;       // opaque VkDescriptorSet
		void* frameSetLayout = nullptr; // opaque VkDescriptorSetLayout for the same set

		// Set 1, always bound: the bindless resource catalog. One instance, never duplicated
		// per frame -- a resource's descriptor is written once at creation and read for the
		// rest of its life.
		void* globalSet = nullptr;       // opaque VkDescriptorSet -- the bindless set
		void* globalSetLayout = nullptr; // opaque VkDescriptorSetLayout for the same set

		// Non-const: Write<K>/WriteSpan<K> below call through it, and a const pointer here never
		// bought anything real -- every prior caller that needed to mutate through it (Node.hpp's
		// PredefinedBufferNode/PredefinedTextureNode/ExecuteOnce, ParticleSystemNode's descriptor
		// refresh) const_cast'd it right back off again.
		ResourceServices* resources = nullptr;

		// True exactly while a dynamic-rendering scope is active for this node (set by
		// PhysicalExecutionBackend::RunSchedule around DynamicRenderingWrapper::Begin/End) --
		// what BeginHostWrite needs to reject an illegal Staged write from inside a render pass
		// with a clear message instead of a cryptic validation error later.
		bool insideRendering = false;

		// This node's own Recipe::domain (set by PhysicalExecutionBackend::RunSchedule, same
		// place as insideRendering above) -- what BeginHostWrite needs to reject a Staged write
		// from a non-Transfer-domain node (see its declaration's comment), and what
		// CheckDeclared below needs to decide throw-vs-warn.
		ExecutionDomain domain = ExecutionDomain::Graphics;

		// This node's own Recipe::realizations for the current frame (set by
		// PhysicalExecutionBackend::RunSchedule/MultiQueueExecutionBackend::RunQueueBatch, same
		// place as domain above) -- the undeclared-access guard's source of truth for "did
		// Setup() actually declare this key." Empty by default (every pre-existing bare-
		// CommandBuffer construction of NodeContext, Graph.hpp's legacy Execute(CommandBuffer&)
		// wrapper included), which CheckDeclared treats as "nothing to check against" rather
		// than "nothing is declared" -- see its own comment.
		std::span<const ResourceRealization> declared{};

		// A node that can't execute concurrently with another has an unexpressed dependency --
		// this is what actually enforces that principle once nodes can be placed on different
		// queues (design plan's ground rules): an access to a key Setup() never declared can't
		// have gotten a barrier/wait synthesized for it, so placement could silently change
		// correctness for exactly the keys this misses. Throws for Compute/Transfer domain
		// (where an undeclared access is a straight-up bug, not just a code-smell -- there's no
		// queue it would ever be *correct* to run on), warns once per K for Graphics (today's
		// only domain with real, audited offenders -- see VolumetricFroxelGenNode's history) so
		// existing passes don't start throwing mid-migration while still surfacing the gap.
		// `declared.empty()` short-circuits before either path: a NodeContext nothing ever
		// populated (the legacy bare-CommandBuffer wrapper, or a test fixture with no recipe in
		// scope at all) has no realizations to check against, which is "unknown," not "none."
		template <typename K>
		void CheckDeclared(const char* accessor) const {
			if (declared.empty()) {
				return;
			}
			const ResourceId id = IdOf<K>();
			for (const ResourceRealization& r : declared) {
				if (r.key == id) {
					return;
				}
			}
			if (domain == ExecutionDomain::Compute || domain == ExecutionDomain::Transfer) {
				throw std::runtime_error(
					std::string("NodeContext::") + accessor + ": undeclared access to '" +
					std::string(id->name) +
					"' from a Compute/Transfer-domain node -- declare it via Read<>/Create<>/"
					"Modify<>() in Setup(), or this node's queue placement could silently change "
					"correctness."
				);
			}
			static bool warned = false;
			if (!warned) {
				std::cerr << "NodeContext::" << accessor << ": undeclared access to '" << id->name
						  << "' from a Graphics-domain node -- declare it via Read<>/Create<>/"
							 "Modify<>() in Setup()."
						  << std::endl;
				warned = true;
			}
		}

		template <typename K>
		[[nodiscard]] std::uint32_t Index() const {
			CheckDeclared<K>("Index");
			return resources ? resources->IndexOf(IdOf<K>()) : 0u;
		}

		template <typename K>
		[[nodiscard]] std::uint32_t StorageIndex(std::uint32_t mip = 0) const {
			CheckDeclared<K>("StorageIndex");
			return resources ? resources->StorageIndexOf(IdOf<K>(), mip) : 0u;
		}

		template <typename K>
		[[nodiscard]] std::uint32_t BufferIndex() const {
			CheckDeclared<K>("BufferIndex");
			return resources ? resources->BufferIndexOf(IdOf<K>()) : 0u;
		}

		// Writes sizeBytes into K's current-frame slot via fn(std::span<std::byte>), then
		// finalizes the write (a real copy command for Staged, a flush for Mapped). A no-op if
		// K isn't host-writable (HostAccess::None) or resources is null -- fn is simply never
		// called in that case, so it's safe to pass a lambda that assumes a full-sized span.
		template <typename K>
		void Write(std::size_t sizeBytes, auto&& fn) {
			CheckDeclared<K>("Write");
			if (!resources) {
				return;
			}
			std::span<std::byte> dst =
				resources->BeginHostWrite(IdOf<K>(), sizeBytes, frameIndex, insideRendering, domain);
			if (dst.empty()) {
				return;
			}
			fn(dst);
			resources->EndHostWrite(IdOf<K>(), cmd);
		}

		// memcpy convenience over Write<K> for the common "I already have the data in one
		// contiguous span" case -- SetData-style callers, mostly. No separate CheckDeclared call
		// here: this always calls Write<K>, which already makes one.
		template <typename K, typename T>
		void WriteSpan(std::span<const T> data) {
			Write<K>(data.size_bytes(), [&](std::span<std::byte> dst) {
				std::memcpy(dst.data(), data.data(), data.size_bytes());
			});
		}
	};

	// Per-frame, backend-agnostic render state (resolution, frame index, ...). Named
	// FrameContext rather than RenderContext: brassica::RenderContext already exists in
	// include/passes/PassResource.hpp and means something different (it carries a live
	// vk::CommandBuffer/VmaAllocator/vk::Device) -- reusing the name would be an ODR
	// violation the moment both headers are included together.
	struct FrameContext {
		std::uint32_t width = 0;
		std::uint32_t height = 0;
		std::uint32_t renderScalePercent = 100;
		std::uint64_t frameIndex = 0;
	};

	// Returned by Node::Setup for a given frame. isActive drives culling; realizations are
	// this node's concrete resource requirements/products for the frame just described by
	// FrameContext.
	struct Recipe {
		ExecutionDomain                  domain = ExecutionDomain::Graphics;
		bool                             isActive = true;
		std::vector<ResourceRealization> realizations{};
	};

	// Barrier metadata -- not a command. A Resource describes how it must be guarded; it
	// never issues anything itself. This is the inversion relative to the existing
	// preRead/preWrite path (src/passes/PassResource.cpp), which emits one pipelineBarrier2
	// per resource per pass with eAllCommands stage masks and therefore can never batch:
	// the resource there issues the command, so there is nothing left for a graph to
	// aggregate. Here the graph is the only thing that can issue anything, so batching is
	// just a matter of merging metadata before flushing it (see BarrierBatch below).
	struct MemoryBarrier {
		ResourceId      resource = nullptr;
		AccessKind      access = AccessKind::Read;
		std::uint32_t   srcStage = 0;
		std::uint32_t   dstStage = 0;
		std::uint32_t   srcAccess = 0;
		std::uint32_t   dstAccess = 0;
		std::uint32_t   oldLayout = 0;
		std::uint32_t   newLayout = 0;
		ExecutionDomain srcDomain = ExecutionDomain::Graphics;
		ExecutionDomain dstDomain = ExecutionDomain::Graphics;
		// No srcQueueFamily/dstQueueFamily (removed, design plan's S4): under
		// VK_SHARING_MODE_CONCURRENT (what every multi-queue-touched resource now uses,
		// PhysicalResource.hpp), a barrier's queue family fields must both be
		// VK_QUEUE_FAMILY_IGNORED -- naming two real, distinct families there is a genuine
		// ownership-transfer request, which CONCURRENT resources never need and never allow.
		// There is no real queue-family ownership transfer in this engine at all: a cross-queue
		// hazard is synchronized by QueueHazardTracker's semaphore wait, not by a barrier.
	};

	class Resource {
	public:
		virtual ~Resource() = default;

		virtual MemoryBarrier GetReadBarrier() const = 0;
		virtual MemoryBarrier GetWriteBarrier() const = 0;
		virtual MemoryBarrier GetReleaseBarrier(ExecutionDomain dstDomain) const = 0;
		virtual MemoryBarrier GetAcquireBarrier(ExecutionDomain srcDomain) const = 0;
	};

	// Accumulates barriers for one scheduled slot and merges them per (resource, srcDomain,
	// dstDomain) so the backend flushes at most one barrier per resource per slot instead of
	// one per read/write call.
	class BarrierBatch {
	public:
		void Add(const MemoryBarrier& b) {
			auto it = std::find_if(m_items.begin(), m_items.end(), [&](const MemoryBarrier& e) {
				return e.resource == b.resource && e.srcDomain == b.srcDomain && e.dstDomain == b.dstDomain;
			});

			if (it == m_items.end()) {
				m_items.push_back(b);
				return;
			}

			it->srcStage |= b.srcStage;
			it->dstStage |= b.dstStage;
			it->srcAccess |= b.srcAccess;
			it->dstAccess |= b.dstAccess;
			it->newLayout = std::max(it->newLayout, b.newLayout);
			if (it->access != b.access) {
				it->access = AccessKind::ReadWrite;
			}
		}

		[[nodiscard]] std::span<const MemoryBarrier> Items() const { return m_items; }

		[[nodiscard]] bool Empty() const { return m_items.empty(); }

		void Clear() { m_items.clear(); }

	private:
		std::vector<MemoryBarrier> m_items;
	};

} // namespace brassica::graph
