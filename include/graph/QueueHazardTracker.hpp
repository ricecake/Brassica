#pragma once
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/ResourceKey.hpp"

// Record-time, state-driven cross-queue hazard detection -- the piece that makes multi-queue
// recording correct regardless of what the compiler's edges do or don't express. Vulkan-free:
// this only ever reasons about abstract per-queue "segment" indices (a segment is one
// contiguous run of commands destined for one real vk::SubmitInfo2 -- see
// MultiQueueExecutionBackend.hpp, the Vulkan-aware owner of segment lifecycle and the only
// caller of this class). A segment index is assigned once, at the moment a queue opens a new
// segment, and is stable from then on; this class never needs to know whether that segment has
// actually been submitted yet, only whether a *target* index has been reached.
//
// Driven by PhysicalResourceRegistry::ResolveId'd resource identity, not compiler edges, because
// the compiler's RAW/WAR/WAW edges miss real hazards by construction: an inactive middle writer
// never gets an edge at all (Graph.hpp's "never synthesize barriers for edges with an inactive
// endpoint"), a zero-init clear isn't a node, and cross-frame carryover on a persistent resource
// has no edge to begin with. Physical state catches all three the same way.
namespace brassica::graph {

	inline constexpr std::size_t kNumGpuQueues = 3; // Graphics, Compute, Transfer -- Host never
													 // records a GPU segment at all.

	inline constexpr std::size_t GpuQueueIndex(ExecutionDomain domain) {
		return static_cast<std::size_t>(domain);
	}

	class QueueHazardTracker {
	public:
		// Every other queue whose last *relevant* touch of `resource` this queue hasn't already
		// waited past. isWriteOrLayoutChange is false only for a plain, same-layout read -- the
		// one access kind that doesn't hazard against another queue's concurrent read (RAR), so
		// it only needs to wait past writes, not reads. Everything else (a write, or a read that
		// changes layout -- a layout transition is itself a write-shaped access) must wait past
		// every other queue's last access of either kind.
		//
		// Pure query: does not itself update any state. The caller (MultiQueueExecutionBackend)
		// is expected to actually satisfy every returned wait, then report back via RecordTouch/
		// MarkCaughtUp below -- this class trusts that contract rather than updating optimistically.
		[[nodiscard]] std::vector<ExecutionDomain>
		QueuesToWaitOn(ResourceId resource, ExecutionDomain queue, bool isWriteOrLayoutChange) const {
			std::vector<ExecutionDomain> result;
			auto                         it = m_resources.find(resource);
			if (it == m_resources.end()) {
				return result; // never touched by anything -- nothing to wait on
			}
			const ResourceHazard& hazard = it->second;
			const std::size_t     q = GpuQueueIndex(queue);

			for (std::size_t p = 0; p < kNumGpuQueues; ++p) {
				if (p == q) {
					continue;
				}
				const std::int32_t relevant = isWriteOrLayoutChange ? hazard.lastAccessSegment[p] : hazard.lastWriteSegment[p];
				if (relevant < 0) {
					continue; // queue p never (relevantly) touched this resource
				}
				if (m_caughtUp[q][p] >= relevant) {
					continue; // already covered by an earlier wait this frame
				}
				result.push_back(DomainOf(p));
			}
			return result;
		}

		// Called once per touch, after the caller has resolved (and satisfied) everything
		// QueuesToWaitOn asked for -- segmentIndex is `queue`'s own currently open segment, i.e.
		// the one this touch's barrier/access is actually being recorded into.
		void RecordTouch(ResourceId resource, ExecutionDomain queue, bool isWriteOrLayoutChange, std::uint32_t segmentIndex) {
			ResourceHazard& hazard = m_resources[resource];
			const std::size_t q = GpuQueueIndex(queue);
			hazard.lastAccessSegment[q] = static_cast<std::int32_t>(segmentIndex);
			if (isWriteOrLayoutChange) {
				hazard.lastWriteSegment[q] = static_cast<std::int32_t>(segmentIndex);
			}
		}

		// Called once per (queue, other) pair the caller actually closed+waited-on to satisfy a
		// QueuesToWaitOn result -- `queue` is now caught up with everything `other` had
		// submitted as of `otherSegmentIndex` (not just the one resource that triggered the
		// wait: a semaphore wait on a whole segment makes everything in it visible, not a single
		// resource's worth).
		void MarkCaughtUp(ExecutionDomain queue, ExecutionDomain other, std::uint32_t otherSegmentIndex) {
			m_caughtUp[GpuQueueIndex(queue)][GpuQueueIndex(other)] = static_cast<std::int32_t>(otherSegmentIndex);
		}

		// Resets per-resource intra-frame hazard state for a new frame -- segment indices start
		// over at 0 next frame regardless of this frame's values (they're frame-local labels,
		// not a global clock). Does *not* preset m_caughtUp: the frame-boundary join (design
		// plan's §Barrier inventory, "Cross-frame" row -- each queue's first segment waits on
		// every other queue's last submitted value from the previous frame) is a real GPU wait
		// the recorder must actually perform, same mechanism as any other wait (MarkCaughtUp
		// after it does) -- presetting m_caughtUp here instead would let this class believe a
		// wait exists that nothing ever actually recorded.
		void BeginFrame() {
			m_resources.clear();
			for (auto& row : m_caughtUp) {
				row.fill(-1);
			}
		}

	private:
		static ExecutionDomain DomainOf(std::size_t queueIndex) { return static_cast<ExecutionDomain>(queueIndex); }

		struct ResourceHazard {
			std::array<std::int32_t, kNumGpuQueues> lastAccessSegment{-1, -1, -1};
			std::array<std::int32_t, kNumGpuQueues> lastWriteSegment{-1, -1, -1};
		};

		std::unordered_map<ResourceId, ResourceHazard>                         m_resources;
		std::array<std::array<std::int32_t, kNumGpuQueues>, kNumGpuQueues>     m_caughtUp{
			{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}
		};
	};

} // namespace brassica::graph
