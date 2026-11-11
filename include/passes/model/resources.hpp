#pragma once

// Model-feature graph resource-key tags -- split out of the old monolithic
// passes/ResourceKeys.hpp (S1, design plan's restructure). The Model*/decor keys below were
// ported forward from that same monolithic file's later (model-work branch) state, once it
// grew the GPU-driven multi-model instancing pipeline -- see ModelRegistry.hpp's own comment
// for the shared-buffer-plus-per-model-base-offset design these all serve.

#include "vulkan/vulkan.hpp"

#include "passes/model/ModelTypes.hpp"
#include "resources/ResourceTypes.hpp"

namespace brassica {

	// Stage 2 (model/ModelRegistry.hpp): one asset's vertex/meshlet data, concatenated across
	// every registered model (glTF-parsed or procedural) into one shared buffer per kind and
	// host-uploaded via graph::PredefinedBufferNode whenever the registry's contents change.
	// ConfiguredExternalBuffer, same reasoning as MaterialBuffer (passes/material/resources.hpp):
	// byte size depends on how many models are registered and how big each one is, so there's no
	// type-level kByteSize constant. Read via a bindless storage-buffer index (ctx.BufferIndex<K>(),
	// binding 6 -- see shaders/model_instance.glsl's own comment) with each model's own
	// ModelDescEntry base added at the lookup point, not a device address -- a multi-model buffer
	// has no single "the" base address a push constant could usefully carry.
	struct GltfVertexBuffer: ConfiguredExternalBuffer<GltfVertexBuffer> {
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};
	struct GltfMeshletVertexBuffer: ConfiguredExternalBuffer<GltfMeshletVertexBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};
	struct GltfMeshletTriangleBuffer: ConfiguredExternalBuffer<GltfMeshletTriangleBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};
	struct GltfMeshletBuffer: ConfiguredExternalBuffer<GltfMeshletBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// GPU-driven model instances (model/ModelInstanceManager.hpp, passes/model/nodes/
	// ModelRenderNode.hpp): the baked-clip successor to the old per-frame-CPU-evaluated
	// GltfJointMatrixBuffer (deleted -- that was one mat4 per joint, re-evaluated and
	// re-uploaded every frame, which doesn't scale to N instances of N models). Palette/clip-
	// table are one-shot PredefinedBufferNode uploads whenever the registry changes. Catalog-
	// converted (ConfiguredExternalBuffer, same reasoning as MaterialBuffer/the four GLTF buffers
	// above): a bare bodyless tag struct no longer compiles as a key at all post-S2b
	// (SetupBuilder::Record<K>() now hard-requires CatalogKey -- see its own comment), so a
	// genuine-runtime-config byte size like this must be published onto the key's own catalog
	// instance (ModelRegistry::RegisterBufferNodes, mirroring MaterialManager::RegisterBufferNode)
	// rather than carried as a bare-struct Create<>() site desc, which no longer exists either.
	struct ModelClipPaletteBuffer: ConfiguredExternalBuffer<ModelClipPaletteBuffer> { // flat glm::mat4[], see ModelClipEntry::paletteBase
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};
	struct ModelClipTableBuffer: ConfiguredExternalBuffer<ModelClipTableBuffer> { // ModelClipEntry[], one entry per clip
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// The GPU-driven instance table itself. Unlike the buffers above, this is genuinely persistent
	// across many frames with no writer at all (sparse CPU writes only on spawn/clip-change/
	// despawn, via ModelInstanceScatterNode) -- kLifetime = Persistent so RelationalCompiler's
	// zero-init-reads never clears it on an idle frame (Create-before-accessor only exempts a key
	// from that if it's flagged Persistent -- see PhysicalExecutionBackend.hpp's
	// ZeroInitializeInactiveReads and RelationalCompiler.hpp's IsPersistentKey). FixedBuffer
	// (not a hand-rolled Desc()): byteSize/usage are both real type-level constants, matching
	// TransmittanceLUT/MultiScatteringLUT/CloudShadowMap's own FixedImage2D<Derived,
	// Lifetime::Persistent> precedent exactly -- none of those append a separate desc.persistent
	// flag either, confirming that's not actually needed alongside kLifetime::Persistent itself.
	struct ModelInstanceBuffer: FixedBuffer<ModelInstanceBuffer, graph::Lifetime::Persistent> {
		static constexpr std::uint64_t kByteSize =
			static_cast<std::uint64_t>(kModelInstanceCapacity) * sizeof(ModelInstance);
		// eTransferSrc (on top of FixedBuffer's implied-by-StorageBufferDesc eStorageBuffer|
		// eTransferDst equivalent, spelled out in full here since FixedBuffer's kUsage is the
		// whole mask, not an addition): lets a debug/test readback (vkCmdCopyBuffer) work,
		// harmless on a persistent buffer this size. No eShaderDeviceAddress: this is read via a
		// bindless storage-buffer index (ctx.BufferIndex<K>(), binding 6), not a device address.
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;
	};

	// Pending sparse writes, coalesced per-slot on the CPU (ModelInstanceManager) and applied to
	// ModelInstanceBuffer by ModelInstanceScatterNode's compute shader. Not persistent: its one
	// real producer (a PredefinedBufferNode) and its one real consumer (the scatter node) are
	// both active only on the same frames (there are pending writes), so Create-before-accessor's
	// ordinary "an active writer precedes this active reader" check always holds whenever this
	// key is actually read -- no idle-frame zero-init gap to guard against, unlike
	// ModelInstanceBuffer above. Still catalog-converted (ConfiguredExternalBuffer) regardless --
	// see ModelClipPaletteBuffer's own comment on why a bare tag struct no longer compiles here
	// at all post-S2b.
	struct ModelInstanceWriteBuffer: ConfiguredExternalBuffer<ModelInstanceWriteBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// Stage 2: one ModelDescEntry per registered model (model/ModelRegistry.hpp), indexed by
	// ModelInstance::modelIndex. Size depends on how many models are registered, so -- same
	// genuine-runtime-config reasoning as ModelClipPaletteBuffer above -- this is catalog-
	// converted (ConfiguredExternalBuffer), not a type-level constant; its real
	// producer (ModelRegistry's own PredefinedBufferNode) publishes the real byteSize onto this
	// key's own catalog instance (ModelRegistry::RegisterBufferNodes).
	struct ModelDescBuffer: ConfiguredExternalBuffer<ModelDescBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// Stage 3 (passes/model/nodes/ModelCullNode.hpp): ModelCullNode's compacted survivor list and
	// its paired indirect task-dispatch args (VkDrawMeshTasksIndirectCommandEXT-shaped:
	// groupCountX/Y/Z), replacing model_render.task's old direct dispatch over every capacity
	// slot. Rebuilt fully every active frame -- same "Create-before-accessor, no stale read
	// possible" reasoning as EntityIndirectBuffer -- no kLifetime::Persistent needed (there's
	// always a same-frame producer whenever either is actually read). FixedBuffer: byte size is
	// capacity-derived, a real type-level constant, not producer-data-dependent.
	struct ModelVisibleListBuffer: FixedBuffer<ModelVisibleListBuffer> {
		static constexpr std::uint64_t kByteSize =
			static_cast<std::uint64_t>(kModelInstanceCapacity) * sizeof(ModelVisibleEntry);
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

	// eIndirectBuffer: consumed directly by drawMeshTasksIndirectEXT. eTransferSrc: lets a
	// debug/test readback (vkCmdCopyBuffer) work.
	struct ModelIndirectArgsBuffer: FixedBuffer<ModelIndirectArgsBuffer> {
		static constexpr std::uint64_t kByteSize = 16; // real struct is 12 bytes; 16 for safety margin
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer |
			vk::BufferUsageFlagBits::eTransferSrc;
	};

	// Stage 4: one ModelSkeletonJoint range per registered model that has skeleton data
	// (ModelAsset::skeleton), indexed by ModelDescEntry::skeletonBase. Same genuine-runtime-config
	// reasoning as ModelDescBuffer above -- size depends on how many models register skeletons,
	// and this is ModelRegistry's own PredefinedBufferNode's sole output.
	struct ModelSkeletonBuffer: ConfiguredExternalBuffer<ModelSkeletonBuffer> {
		static constexpr vk::BufferUsageFlags kUsage = GltfVertexBuffer::kUsage;
	};

	// Stage 4 (passes/model/nodes/ModelCullNode.hpp's live-mode claim, passes/model/nodes/
	// ModelLiveEvalNode.hpp): the compacted list of instances ModelCullNode claimed a live-pool
	// slot for this frame (flat uint instanceIndex[], one per claimed slot -- the slot index IS
	// the claim order, so no separate slot field is needed), its indirect-dispatch args, and the
	// transient live-palette pool ModelLiveEvalNode writes into. All three are FixedBuffer
	// (fixed-budget sizing, same reasoning as ModelVisibleListBuffer/ModelIndirectArgsBuffer
	// above) and rebuilt fully every active frame -- no kLifetime::Persistent needed, same "no
	// stale read possible" argument.
	struct ModelLiveListBuffer: FixedBuffer<ModelLiveListBuffer> {
		static constexpr std::uint64_t kByteSize = static_cast<std::uint64_t>(kModelLiveBudget) * 4u;
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

	// {groupCountX, groupCountY, groupCountZ, rawLiveCount} -- 16 bytes. groupCountX/Y/Z are
	// ModelLiveEvalNode's own indirect-dispatch args (64-wide workgroups, so groupCountX is
	// ceil(rawLiveCount/64), NOT rawLiveCount itself -- unlike ModelIndirectArgsBuffer's
	// one-workgroup-per-entry draw, this is a plain many-threads-per-workgroup compute dispatch,
	// so the raw atomic claim count needs a real divide step before it's a valid workgroup
	// count). rawLiveCount is ModelCullNode's own atomic claim counter (word offset 12); a tiny
	// dedicated ModelLiveFinalizeNode does that divide between ModelCullNode and
	// ModelLiveEvalNode, same "let the graph insert the barrier between distinct nodes" reasoning
	// as the Reset/Cull split above -- not a hand-rolled intra-node barrier.
	struct ModelLiveIndirectArgsBuffer: FixedBuffer<ModelLiveIndirectArgsBuffer> {
		static constexpr std::uint64_t kByteSize = 16;
		static constexpr vk::BufferUsageFlags kUsage = vk::BufferUsageFlagBits::eStorageBuffer |
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer;
	};

	// Flat mat4[kModelLiveBudget * kModelLiveMaxJointCount], addressed as
	// liveSlot * kModelLiveMaxJointCount + joint -- see ModelVisibleEntry's own comment on how a
	// live-mode entry's paletteBaseA/B point in here instead of ModelClipPaletteBuffer.
	struct ModelLivePaletteBuffer: FixedBuffer<ModelLivePaletteBuffer> {
		static constexpr std::uint64_t kByteSize =
			static_cast<std::uint64_t>(kModelLiveBudget) * kModelLiveMaxJointCount * sizeof(glm::mat4);
		static constexpr vk::BufferUsageFlags kUsage =
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	};

} // namespace brassica
