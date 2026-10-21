#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/ResourceState.hpp"
#include "graph/VulkanSeam.hpp"

namespace brassica::graph {

	// Which half of a producer/consumer handoff a BarrierBatch represents. Acquire covers
	// everything that transitions a resource into the layout its next access needs: first-use
	// transitions, same-domain edges, and the consumer side of a cross-domain edge. Release is
	// the producer side of a cross-domain edge -- see DispatchRelease for why that side
	// deliberately carries no layout or per-resource specifics at all.
	enum class BarrierPhase : std::uint8_t { Acquire, Release };

	class BarrierTranslator {
	public:
		// registry is non-const: this is the one place tracked resource state (PhysicalTexture/
		// PhysicalBuffer's current layout/stage/access -- see PhysicalResource.hpp) actually
		// changes. DynamicRenderingWrapper (PhysicalExecutionBackend.hpp) only ever reads it
		// through a const registry reference; that asymmetry is deliberate, not an oversight --
		// see PhysicalRegistry.hpp's non-const GetTexture/GetBuffer overloads.
		//
		// currentQueue: nullopt (the default, every existing call site) preserves today's
		// behavior exactly -- a resource's queue-touch mask is never read, so no barrier's src is
		// ever sanitized away from tracked state. Pass the real domain this batch is being
		// recorded for only from a multi-queue caller that has *actually* split batches by
		// destination domain into separate per-queue command buffers (QueueHazardTracker.hpp) --
		// passing a real value here while still recording a mixed-domain batch into one shared
		// command buffer would wrongly sanitize same-queue barriers that have nothing to do with
		// a real queue boundary.
		static void TranslateAndDispatch(
			vk::CommandBuffer              cmd,
			PhysicalResourceRegistry&      registry,
			const BarrierBatch&            batch,
			BarrierPhase                   phase,
			std::optional<ExecutionDomain> currentQueue = std::nullopt
		) {
			if (batch.Empty() || !cmd) {
				return;
			}

			if (phase == BarrierPhase::Release) {
				DispatchRelease(cmd, registry, batch);
			} else {
				DispatchAcquire(cmd, registry, batch, currentQueue);
			}
		}

	private:
		// Read < Write < ReadWrite -- used only to deterministically pick a winner when two
		// batch entries for the same resource disagree on access (see the coalescing comment in
		// DispatchAcquire). Not a correctness claim: a genuine same-stage conflict between two
		// producers wanting different write layouts is a graph bug that Provision() detects
		// separately (PhysicalRegistry.hpp's same-stage conflicting-layout check).
		static int AccessPriority(AccessKind access) {
			switch (access) {
			case AccessKind::Read:
				return 0;
			case AccessKind::Write:
				return 1;
			case AccessKind::ReadWrite:
				return 2;
			}
			return 0;
		}

		// One resolved barrier target after coalescing every batch entry touching the same
		// resource. BarrierBatch::Add merges by (resource, srcDomain, dstDomain) (Execution.hpp),
		// so e.g. two producers on different domains into one consumer can still leave two
		// entries for the same resource in one batch -- this reduces those to one barrier.
		struct Coalesced {
			std::shared_ptr<PhysicalTexture>               tex;
			std::shared_ptr<PhysicalBuffer>                buf;
			std::shared_ptr<PhysicalAccelerationStructure> as;
			vk::ImageLayout                                layout{}; // only meaningful when tex is set
			vk::PipelineStageFlags2                        stage{};
			vk::AccessFlags2                               access{};
			int                                            priority = -1;
		};

		static void DispatchAcquire(
			vk::CommandBuffer               cmd,
			PhysicalResourceRegistry&       registry,
			const BarrierBatch&             batch,
			std::optional<ExecutionDomain>  currentQueue
		) {
			std::unordered_map<ResourceId, Coalesced> coalesced;

			for (const auto& mb : batch.Items()) {
				auto tex = registry.GetTexture(mb.resource);
				auto buf = tex ? nullptr : registry.GetBuffer(mb.resource);
				auto as = (tex || buf) ? nullptr : registry.GetAccelerationStructure(mb.resource);
				if (!tex && !buf && !as) {
					// A History<K> whose target key was never actually provisioned under its own
					// id is the known case this catches (see the barrier migration plan's Known
					// Limitations list) -- silently dropping the barrier used to be how that hid.
					throw std::runtime_error(
						"BarrierTranslator: barrier resource '" + std::string(mb.resource->name) +
						"' resolved to neither a texture, a buffer, nor an acceleration structure in "
						"the registry"
					);
				}

				ResourceState derived;
				if (tex) {
					derived = DeriveImageState(
						mb.access,
						mb.dstDomain,
						tex->GetDesc().usageMask ? vk::ImageUsageFlags(tex->GetDesc().usageMask)
												 : vk::ImageUsageFlags{},
						static_cast<vk::Format>(tex->GetDesc().formatCode)
					);
				} else if (buf) {
					derived = DeriveBufferState(
						mb.access,
						mb.dstDomain,
						buf->GetDesc().usageMask ? vk::BufferUsageFlags(buf->GetDesc().usageMask)
												 : vk::BufferUsageFlags{}
					);
				} else {
					derived = DeriveAccelerationStructureState(mb.access, mb.dstDomain);
				}

				// Keyed by the resolved id (PhysicalResourceRegistry::ResolveId), not mb.resource
				// directly -- see that function's own comment for why this indirection still exists.
				auto& entry = coalesced[registry.ResolveId(mb.resource)];
				if (!entry.tex && !entry.buf && !entry.as) {
					entry.tex = tex;
					entry.buf = buf;
					entry.as = as;
				}
				const int priority = AccessPriority(mb.access);
				if (priority > entry.priority) {
					entry.layout = derived.layout;
					entry.priority = priority;
				}
				entry.stage |= derived.stage;
				entry.access |= derived.access;
			}

			std::vector<vk::ImageMemoryBarrier2>  imageBarriers;
			std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
			std::vector<vk::MemoryBarrier2>       memoryBarriers;

			for (auto& [resource, entry] : coalesced) {
				if (entry.tex) {
					EmitImageBarrier(*entry.tex, entry.layout, entry.stage, entry.access, imageBarriers, currentQueue);
				} else if (entry.buf) {
					EmitBufferBarrier(*entry.buf, entry.stage, entry.access, bufferBarriers, currentQueue);
				} else {
					EmitAccelerationStructureBarrier(*entry.as, entry.stage, entry.access, memoryBarriers, currentQueue);
				}
			}

			Flush(cmd, imageBarriers, bufferBarriers, memoryBarriers);
		}

		// True iff `mask` (a resource's queue-touch mask, PhysicalResource.hpp) contains a bit
		// other than `currentQueue`'s own -- i.e. some other GPU queue contributed to this
		// resource's tracked lastStage/lastAccess since currentQueue itself last did, whether via
		// a real barrier or an accumulated read-after-read skip. currentQueue == nullopt (no
		// multi-queue caller in the picture) always reads as "not cross-queue" -- see
		// TranslateAndDispatch's currentQueue comment for why that's the correct default.
		static bool IsCrossQueue(std::uint8_t mask, std::optional<ExecutionDomain> currentQueue) {
			return currentQueue.has_value() && (mask & ~QueueBit(*currentQueue)) != 0;
		}

		// Skip rule: only a genuine read-after-read (same layout, neither side a write) may be
		// skipped, and even then the resource's tracked stage/access is still accumulated rather
		// than left untouched -- otherwise a later write's srcStage/srcAccess would only see the
		// most recent read it happened to barrier against, not every read since the last write
		// (a WAR hazard). Everything else always emits.
		//
		// This one rule also covers what might look like it needs a special case: "an edge-
		// derived entry must always emit, never be skipped by layout equality." It doesn't need
		// one. CollectEdges (Graph.hpp) only ever connects a producer that writes the key to a
		// consumer, and leveling guarantees the producer's stage strictly precedes the
		// consumer's -- so by the time an edge's barrier reaches this function, tracked.access is
		// always the producer's write, IsWrite(tracked.access) is always true, and the skip
		// condition's `!IsWrite(tracked.access)` is always false. Edges always emit as a
		// structural consequence, not a hand-coded exception.
		//
		// A cross-queue read-after-read is still a legal skip: neither side writes, so there is
		// no hazard to synchronize regardless of which queue(s) did the reading (the hazard
		// table that decided whether a *wait* was needed already covers that -- see
		// QueueHazardTracker.hpp). The mask is still OR-accumulated across both queues below, same
		// as the same-queue case, so a *later* write sees every reader it must have waited past.
		// No srcQueueFamily/dstQueueFamily param -- removed, design plan's S4. A barrier never
		// names real queue families here: there's no real ownership transfer in this engine at
		// all (every multi-queue-touched resource is VK_SHARING_MODE_CONCURRENT,
		// PhysicalResource.hpp, which forbids naming one), so the real
		// vk::ImageMemoryBarrier2/vk::BufferMemoryBarrier2 below always passes
		// VK_QUEUE_FAMILY_IGNORED for both. The RAR-skip rule's old "same family, or one side
		// ignored" clause is gone with it -- IsCrossQueue (queue-touch-mask-based, below) is the
		// one and only cross-queue signal this function needs now.
		static void EmitImageBarrier(
			PhysicalTexture&                      tex,
			vk::ImageLayout                       newLayout,
			vk::PipelineStageFlags2               dstStage,
			vk::AccessFlags2                      dstAccess,
			std::vector<vk::ImageMemoryBarrier2>& out,
			std::optional<ExecutionDomain>        currentQueue = std::nullopt
		) {
			const bool readAfterRead =
				tex.GetCurrentLayout() == newLayout && !IsWrite(tex.GetLastAccess()) && !IsWrite(dstAccess);
			if (readAfterRead) {
				tex.SetLastStageAccess(tex.GetLastStage() | dstStage, tex.GetLastAccess() | dstAccess);
				if (currentQueue.has_value()) {
					tex.SetQueueTouchMask(tex.GetQueueTouchMask() | QueueBit(*currentQueue));
				}
				return;
			}

			// Cross-queue: tex.GetLastStage()/GetLastAccess() describe what some *other* queue
			// did, which a pipeline barrier on currentQueue's own command buffer cannot
			// meaningfully cite as its src -- barriers only order commands already submitted to
			// the same queue. The hazard table (QueueHazardTracker.hpp) has already inserted
			// whatever semaphore wait makes that other queue's work visible-before this point;
			// eAllCommands/no-access-bits here just orders this barrier after currentQueue's own
			// prior work (trivially true if there is none yet) without claiming a src scope this
			// queue never actually produced.
			const bool crossQueue = IsCrossQueue(tex.GetQueueTouchMask(), currentQueue);

			const auto& desc = tex.GetDesc();

			out.push_back(
				vk::ImageMemoryBarrier2{
					crossQueue ? vk::PipelineStageFlagBits2::eAllCommands
							   : (tex.GetLastStage() ? tex.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe),
					crossQueue ? vk::AccessFlags2{} : tex.GetLastAccess(),
					dstStage,
					dstAccess,
					tex.GetCurrentLayout(),
					newLayout,
					VK_QUEUE_FAMILY_IGNORED,
					VK_QUEUE_FAMILY_IGNORED,
					tex.GetImage(),
					vk::ImageSubresourceRange(
						AspectFor(static_cast<vk::Format>(desc.formatCode)),
						0,
						desc.mips ? desc.mips : 1,
						0,
						desc.layers ? desc.layers : 1
					),
				}
			);
			tex.SetCurrentLayout(newLayout);
			tex.SetLastStageAccess(dstStage, dstAccess);
			if (currentQueue.has_value()) {
				tex.SetQueueTouchMask(QueueBit(*currentQueue));
			}
		}

		static void EmitBufferBarrier(
			PhysicalBuffer&                        buf,
			vk::PipelineStageFlags2                dstStage,
			vk::AccessFlags2                       dstAccess,
			std::vector<vk::BufferMemoryBarrier2>& out,
			std::optional<ExecutionDomain>         currentQueue = std::nullopt
		) {
			const bool readAfterRead = !IsWrite(buf.GetLastAccess()) && !IsWrite(dstAccess);
			if (readAfterRead) {
				buf.SetLastStageAccess(buf.GetLastStage() | dstStage, buf.GetLastAccess() | dstAccess);
				if (currentQueue.has_value()) {
					buf.SetQueueTouchMask(buf.GetQueueTouchMask() | QueueBit(*currentQueue));
				}
				return;
			}

			// See EmitImageBarrier's identical comment just above.
			const bool crossQueue = IsCrossQueue(buf.GetQueueTouchMask(), currentQueue);

			const auto& desc = buf.GetDesc();

			out.push_back(
				vk::BufferMemoryBarrier2{
					crossQueue ? vk::PipelineStageFlagBits2::eAllCommands
							   : (buf.GetLastStage() ? buf.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe),
					crossQueue ? vk::AccessFlags2{} : buf.GetLastAccess(),
					dstStage,
					dstAccess,
					VK_QUEUE_FAMILY_IGNORED,
					VK_QUEUE_FAMILY_IGNORED,
					buf.GetBuffer(),
					0,
					desc.byteSize ? desc.byteSize : VK_WHOLE_SIZE,
				}
			);
			buf.SetLastStageAccess(dstStage, dstAccess);
			if (currentQueue.has_value()) {
				buf.SetQueueTouchMask(QueueBit(*currentQueue));
			}
		}

		// Per the Vulkan spec, an acceleration structure is synchronized via a generic,
		// handle-free vk::MemoryBarrier2 -- there is no ImageMemoryBarrier2/BufferMemoryBarrier2
		// equivalent that names the AS itself (see PhysicalAccelerationStructure's class comment,
		// PhysicalResource.hpp). Same skip rule as EmitBufferBarrier: no layout to compare, so a
		// read-after-read is just "neither side is a write."
		static void EmitAccelerationStructureBarrier(
			PhysicalAccelerationStructure&   as,
			vk::PipelineStageFlags2          dstStage,
			vk::AccessFlags2                 dstAccess,
			std::vector<vk::MemoryBarrier2>& out,
			std::optional<ExecutionDomain>   currentQueue = std::nullopt
		) {
			const bool readAfterRead = !IsWrite(as.GetLastAccess()) && !IsWrite(dstAccess);
			if (readAfterRead) {
				as.SetLastStageAccess(as.GetLastStage() | dstStage, as.GetLastAccess() | dstAccess);
				if (currentQueue.has_value()) {
					as.SetQueueTouchMask(as.GetQueueTouchMask() | QueueBit(*currentQueue));
				}
				return;
			}

			// See EmitImageBarrier's identical comment on why cross-queue sanitizes src.
			const bool crossQueue = IsCrossQueue(as.GetQueueTouchMask(), currentQueue);

			out.push_back(
				vk::MemoryBarrier2{
					crossQueue ? vk::PipelineStageFlagBits2::eAllCommands
							   : (as.GetLastStage() ? as.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe),
					crossQueue ? vk::AccessFlags2{} : as.GetLastAccess(),
					dstStage,
					dstAccess,
				}
			);
			as.SetLastStageAccess(dstStage, dstAccess);
			if (currentQueue.has_value()) {
				as.SetQueueTouchMask(QueueBit(*currentQueue));
			}
		}

		// The producer side of a cross-domain edge (Graph.hpp's SynthesizeBarrier posts these to
		// the producer's own stage). For most resources deriving a layout here would be
		// meaningless -- it would come from the producer's access and the *consumer's* domain,
		// which describes neither side correctly -- and it would double-transition the resource,
		// since the edge's Acquire-phase entry in the consumer's stage already performs the real
		// transition. Worse, one of these can legitimately fire while a resource is still
		// eUndefined: Import<K> nodes run in ExecutionDomain::Host with no realizations of their
		// own (Node.hpp), so a Graphics consumer of an imported key produces a Host->Graphics
		// edge whose release half sits in the Host stage before anything has touched the
		// resource. A per-image barrier there would be newLayout=eUndefined -- exactly Bug 1 this
		// rewrite exists to fix. A single global memory barrier has no layout to get wrong, which
		// is why that stays the default for everything below.
		//
		// The one exception (design plan's S5, "producer-side pre-transition"): when every real
		// consumer of a given (producer, resource) pair is a plain Read, and every one of those
		// consumers' own domains derives the *same* target layout (checked below by actually
		// deriving it per domain, never assumed), this resource's transition can happen exactly
		// once, right here, instead of once per consumer's own acquire-phase barrier -- which is
		// what forces N independent consumers on N different queues to serialize against each
		// other over who performs that same transition first (test_multiqueue_sync.cpp's case
		// (b)). Graph.hpp's SynthesizeBarrier stores the *consumer's* access (not the producer's)
		// in each post-barrier entry specifically so this grouping can tell; see its own comment.
		// Still guarded by GetLastStage() being non-empty -- the exact same "never touched yet"
		// signal that rules out the Import<K>/eUndefined case above, since a real per-resource
		// transition is exactly the thing that case must never get.
		//
		// There is also no real queue-family ownership transfer happening here to "release":
		// VK_QUEUE_FAMILY_IGNORED everywhere, one graphicsQueue in the whole engine. This is
		// future work behind an ExecutionDomain -> queue-family mapping, not attempted here.
		static void
		DispatchRelease(vk::CommandBuffer cmd, PhysicalResourceRegistry& registry, const BarrierBatch& batch) {
			vk::PipelineStageFlags2 srcStage{};
			vk::AccessFlags2        srcAccess{};

			// Grouped by resolved resource, not iterated per-entry directly: a resource read by
			// more than one distinct consumer domain (e.g. both a Compute and a Transfer reader)
			// posts one post-barrier entry per distinct (resource, dstDomain) pair (BarrierBatch::
			// Add's own dedup key) -- the pre-transition decision needs every one of those
			// domains together, not just whichever entry is seen first.
			struct ImageReleaseGroup {
				std::shared_ptr<PhysicalTexture> tex;
				std::vector<ExecutionDomain>     consumerDomains;
				bool                             allRead = true;
			};
			std::unordered_map<ResourceId, ImageReleaseGroup> imageGroups;

			for (const auto& mb : batch.Items()) {
				if (auto tex = registry.GetTexture(mb.resource)) {
					auto& group = imageGroups[registry.ResolveId(mb.resource)];
					group.tex = tex;
					group.consumerDomains.push_back(mb.dstDomain);
					if (mb.access != AccessKind::Read) {
						group.allRead = false;
					}
					continue;
				}
				if (auto buf = registry.GetBuffer(mb.resource)) {
					srcStage |= buf->GetLastStage();
					srcAccess |= buf->GetLastAccess();
				}
				// Unlike DispatchAcquire, an unresolvable resource here isn't thrown on: the
				// release barrier is best-effort ordering for whatever this stage did, not a
				// per-resource correctness statement, and the Acquire-phase pass over the same
				// batch already throws on this exact case.
			}

			std::vector<vk::ImageMemoryBarrier2> preTransitionBarriers;
			for (auto& [resolved, group] : imageGroups) {
				PhysicalTexture& tex = *group.tex;
				if (!group.allRead || !tex.GetLastStage()) {
					srcStage |= tex.GetLastStage();
					srcAccess |= tex.GetLastAccess();
					continue;
				}

				const auto&               desc = tex.GetDesc();
				const vk::ImageUsageFlags usage =
					desc.usageMask ? vk::ImageUsageFlags(desc.usageMask) : vk::ImageUsageFlags{};
				const vk::Format format = static_cast<vk::Format>(desc.formatCode);

				const vk::ImageLayout agreedLayout =
					DeriveImageState(AccessKind::Read, group.consumerDomains.front(), usage, format).layout;
				vk::AccessFlags2 dstAccess{};
				bool             allAgree = true;
				for (ExecutionDomain d : group.consumerDomains) {
					const ResourceState derived = DeriveImageState(AccessKind::Read, d, usage, format);
					if (derived.layout != agreedLayout) {
						allAgree = false;
						break;
					}
					dstAccess |= derived.access;
				}
				if (!allAgree) {
					srcStage |= tex.GetLastStage();
					srcAccess |= tex.GetLastAccess();
					continue;
				}

				// dstStage is eAllCommands unconditionally here, not each consumer domain's own
				// derived (and OR'd) stage: DeriveImageState's per-domain stage is only ever
				// meant to pair with that *same* domain's own derived access in one real barrier
				// (e.g. a Transfer-domain Sampled-Read derives ShaderStageForDomain(Transfer)
				// paired with eShaderSampledRead -- individually harmless under the ordinary RAR-
				// skip path, which never builds a real barrier from it, but combining several
				// domains' stages and access bits together here can produce a real, spec-invalid
				// pairing no single domain would ever actually hit on its own). eAllCommands is
				// always a valid dstStage for any access mask the Vulkan spec defines, and this
				// barrier's only real job is the layout transition itself, not minimal stage
				// scoping.
				const vk::PipelineStageFlags2 dstStage = vk::PipelineStageFlagBits2::eAllCommands;

				preTransitionBarriers.push_back(
					vk::ImageMemoryBarrier2{
						tex.GetLastStage(),
						tex.GetLastAccess(),
						dstStage,
						dstAccess,
						tex.GetCurrentLayout(),
						agreedLayout,
						VK_QUEUE_FAMILY_IGNORED,
						VK_QUEUE_FAMILY_IGNORED,
						tex.GetImage(),
						vk::ImageSubresourceRange(
							AspectFor(format),
							0,
							desc.mips ? desc.mips : 1,
							0,
							desc.layers ? desc.layers : 1
						),
					}
				);
				tex.SetCurrentLayout(agreedLayout);
				tex.SetLastStageAccess(dstStage, dstAccess);
			}

			if (!srcStage && !srcAccess && preTransitionBarriers.empty()) {
				return;
			}

			std::vector<vk::MemoryBarrier2> memoryBarriers;
			if (srcStage || srcAccess) {
				memoryBarriers.push_back(
					vk::MemoryBarrier2{
						srcStage ? srcStage : vk::PipelineStageFlagBits2::eTopOfPipe,
						srcAccess,
						vk::PipelineStageFlagBits2::eAllCommands,
						vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
					}
				);
			}

			vk::DependencyInfo depInfo{};
			depInfo.setMemoryBarriers(memoryBarriers);
			depInfo.setImageMemoryBarriers(preTransitionBarriers);
			cmd.pipelineBarrier2(depInfo);
		}

		static void Flush(
			vk::CommandBuffer                      cmd,
			std::vector<vk::ImageMemoryBarrier2>&  imageBarriers,
			std::vector<vk::BufferMemoryBarrier2>& bufferBarriers,
			std::vector<vk::MemoryBarrier2>&       memoryBarriers
		) {
			if (imageBarriers.empty() && bufferBarriers.empty() && memoryBarriers.empty()) {
				return;
			}

			vk::DependencyInfo depInfo{
				{},
				static_cast<std::uint32_t>(memoryBarriers.size()),
				memoryBarriers.empty() ? nullptr : memoryBarriers.data(),
				static_cast<std::uint32_t>(bufferBarriers.size()),
				bufferBarriers.empty() ? nullptr : bufferBarriers.data(),
				static_cast<std::uint32_t>(imageBarriers.size()),
				imageBarriers.empty() ? nullptr : imageBarriers.data(),
			};
			cmd.pipelineBarrier2(depInfo);
		}
	};

} // namespace brassica::graph
