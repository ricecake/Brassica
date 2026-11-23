#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "graph/Node.hpp"
#include "passes/model/ModelTypes.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/model/resources.hpp"

namespace brassica::graph {
	class Graph;
	class ResourceCatalog;
}

namespace brassica {

	using ModelInstanceHandle = std::uint32_t;
	inline constexpr ModelInstanceHandle kInvalidModelInstanceHandle = 0xFFFFFFFFu;

	// The only path onto/off of the GPU-driven instance table (ModelInstanceBuffer):
	// Spawn/SetClip/SetTransform/Despawn edit a CPU-side mirror (m_records, authoritative since
	// the CPU is the table's only writer) and mark the touched slot dirty; RegisterBufferNodes
	// flushes whatever's dirty through ModelInstanceWriteBuffer once per frame.
	// ModelInstanceScatterNode (passes/ModelInstanceScatterNode.hpp) applies those writes to
	// ModelInstanceBuffer on the GPU -- the CPU never touches that buffer directly. On a frame
	// with nothing dirty, nothing is uploaded and the scatter node goes inactive, matching the
	// plan's "a fully GPU-driven instance never has to be communicated to the CPU explicitly
	// unless specifically needed."
	//
	// Stage 5.2: this only ever owns slots [0, kManualInstanceCapacity) -- the default
	// constructor argument, not kModelInstanceCapacity (the *total* buffer size). The remaining
	// [kDecorSlotBase, kModelInstanceCapacity) range is DecorChunkAllocator's, written directly
	// from the GPU, never through this class's free list at all. Two disjoint ranges in one
	// shared buffer, not two buffers -- see ModelTypes.hpp's own comment on why.
	class ModelInstanceManager {
	public:
		explicit ModelInstanceManager(std::uint32_t capacity = kManualInstanceCapacity):
			m_records(capacity), m_capacity(capacity) {
			m_freeSlots.reserve(capacity);
			for (std::uint32_t slot = capacity; slot-- > 0;) {
				m_freeSlots.push_back(slot); // reverse order: slot 0 pops first
			}

			// ModelInstanceWriteBuffer needs a real (non-empty) desc from the very first frame
			// this is ever registered, even before anything has spawned. PredefinedBufferNode's
			// own `dirty` starts true regardless of whether SetData has ever been called with
			// real content -- so without this, frame 1's Setup()/Execute() run unconditionally
			// (Active(everyFrame || dirty) is true) against a still-default-constructed empty
			// ResourceDesc, which Provision() correctly ignores ("an empty desc is simply
			// ignored for provisioning purposes") -- but then leaves nothing in the registry for
			// that frame's barrier synthesis to resolve, producing BarrierTranslator's "resolved
			// to neither a texture, a buffer, nor an acceleration structure." A real, confirmed
			// startup crash, not hypothetical. The real desc now lives on ModelInstanceWriteBuffer's
			// own catalog instance (RegisterBufferNodes, published every frame, fixed at this
			// constructor's capacity -- never size-derived from the write count, so every frame
			// publishes the exact same byteSize); SetData() here is just the data assignment.
			m_writeBufNode.SetData({});
		}

		// Returns kInvalidModelInstanceHandle if capacity is exhausted. New instances start at
		// the rest pose (kModelClipRestPose) -- SetClip is a separate, optional call, same
		// "nothing is communicated unless specifically needed" reasoning as the class comment.
		// modelIndex is explicit, not defaulted: Stage 2's registry can hold several producers'
		// models (model/ModelRegistry.hpp), and a silently-defaulted 0 would mask a caller that
		// forgot to pick one. liveCapable sets Stage 4's kModelInstanceLiveCapableBit -- a
		// persistent property the caller opts into (e.g. "this is a tree instance I want wind
		// on"), not the per-frame mode ModelCullNode actually picks (which also checks the
		// model's own skeleton presence and distance/pool budget, so setting this on a
		// non-skeletal model's instance just safely falls back to baked, see ModelDescEntry's
		// own comment on skeletonBase).
		[[nodiscard]] ModelInstanceHandle Spawn(
			std::uint32_t modelIndex,
			glm::vec3     position,
			float         scale = 1.0f,
			glm::vec4     rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
			bool          liveCapable = false
		) {
			if (m_freeSlots.empty()) {
				return kInvalidModelInstanceHandle;
			}
			std::uint32_t slot = m_freeSlots.back();
			m_freeSlots.pop_back();

			ModelInstance& record = m_records[slot];
			record = ModelInstance{};
			record.modelIndex = modelIndex;
			record.position = position;
			record.scale = scale;
			record.rotation = rotation;
			record.clipIndex = kModelClipRestPose;
			record.flags = kModelInstanceAliveBit | (liveCapable ? kModelInstanceLiveCapableBit : 0u);
			// Cheap deterministic per-slot hash (Knuth multiplicative) for ModelLiveEvalNode's
			// seed-bend/wind-gust phase diversity -- not cryptographic, just needs to decorrelate
			// adjacent slots so a herd doesn't sway in lockstep.
			record.seed = slot * 2654435761u;
			m_dirtySlots.insert(slot);
			return slot;
		}

		void SetClip(ModelInstanceHandle handle, std::uint32_t clipIndex, std::uint32_t startTimeMs, bool shouldLoop) {
			if (handle >= m_capacity) {
				return;
			}
			ModelInstance& record = m_records[handle];
			record.clipIndex = clipIndex;
			record.startTimeMs = startTimeMs;
			record.flags = kModelInstanceAliveBit | (shouldLoop ? kModelInstanceLoopBit : 0u);
			m_dirtySlots.insert(handle);
		}

		void SetTransform(ModelInstanceHandle handle, glm::vec3 position, glm::vec4 rotation, float scale) {
			if (handle >= m_capacity) {
				return;
			}
			ModelInstance& record = m_records[handle];
			record.position = position;
			record.rotation = rotation;
			record.scale = scale;
			m_dirtySlots.insert(handle);
		}

		void Despawn(ModelInstanceHandle handle) {
			if (handle >= m_capacity) {
				return;
			}
			m_records[handle].flags &= ~kModelInstanceAliveBit;
			m_dirtySlots.insert(handle);
			m_freeSlots.push_back(handle);
		}

		[[nodiscard]] std::uint32_t Capacity() const { return m_capacity; }

		// Lets callers (e.g. the debug UI widget) pick a sensible startTimeMs -- SetClip itself
		// still takes it explicitly (callers that want a deliberate offset, like desyncing a
		// herd's phases, need that), this is purely a convenience for "now." Updated once per
		// frame from Engine.cpp, same call site as RegisterBufferNodes.
		void SetCurrentTimeMs(std::uint32_t timeMs) { m_currentTimeMs = timeMs; }

		[[nodiscard]] std::uint32_t GetCurrentTimeMs() const { return m_currentTimeMs; }

		// How many writes RegisterBufferNodes just flushed to ModelInstanceWriteBuffer this
		// frame -- ModelInstanceScatterNode (queried via ServiceLocator, same indirection
		// IFoliageManager/IParticleManager use) reads this in its own Setup()/Execute() to size
		// its dispatch and decide activity. Valid for the rest of this frame: Engine.cpp calls
		// RegisterBufferNodes before any node's Setup() runs.
		[[nodiscard]] std::uint32_t GetLastFlushCount() const { return m_lastFlushCount; }

		// Same per-frame-rebuild shape as ModelRegistry::RegisterBufferNodes -- Engine.cpp
		// rebuilds graph::Graph fresh every frame, so this needs calling every frame regardless
		// of whether anything is actually dirty (the underlying PredefinedBufferNode's own
		// dirty flag is what decides whether a real upload happens this frame). Publishes
		// ModelInstanceWriteBuffer's byteSize onto its own catalog instance first -- see the
		// constructor's own comment on why it must stay this fixed capacity, not size-derived
		// from the write count, on every call including the very first.
		void RegisterBufferNodes(graph::Graph& frameGraph, graph::ResourceCatalog& catalog);

	private:
		std::vector<ModelInstance>        m_records;
		std::vector<std::uint32_t>        m_freeSlots;
		std::unordered_set<std::uint32_t> m_dirtySlots;
		std::uint32_t                     m_capacity;
		std::uint32_t                     m_lastFlushCount{0};
		std::uint32_t                     m_currentTimeMs{0};

		graph::PredefinedBufferNode<ModelInstanceWriteBuffer, ModelInstanceWrite, SubPhase::Prepare> m_writeBufNode;
	};

} // namespace brassica
