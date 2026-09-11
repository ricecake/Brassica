#pragma once

#include <cstdint>
#include <memory>
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
		static void TranslateAndDispatch(
			vk::CommandBuffer         cmd,
			PhysicalResourceRegistry& registry,
			const BarrierBatch&       batch,
			BarrierPhase              phase
		) {
			if (batch.Empty() || !cmd) {
				return;
			}

			if (phase == BarrierPhase::Release) {
				DispatchRelease(cmd, registry, batch);
			} else {
				DispatchAcquire(cmd, registry, batch);
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

		static void
		DispatchAcquire(vk::CommandBuffer cmd, PhysicalResourceRegistry& registry, const BarrierBatch& batch) {
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

				// Keyed by the resolved id, not mb.resource directly: two different versions of one
				// key (VersionedKey<K,1>, VersionedKey<K,2>) name the same PhysicalTexture, and must
				// coalesce into one barrier against it rather than two EmitImageBarrier calls racing
				// to mutate the same tracked state.
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
					EmitImageBarrier(*entry.tex, entry.layout, entry.stage, entry.access, imageBarriers);
				} else if (entry.buf) {
					EmitBufferBarrier(*entry.buf, entry.stage, entry.access, bufferBarriers);
				} else {
					EmitAccelerationStructureBarrier(*entry.as, entry.stage, entry.access, memoryBarriers);
				}
			}

			Flush(cmd, imageBarriers, bufferBarriers, memoryBarriers);
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
		static void EmitImageBarrier(
			PhysicalTexture&                      tex,
			vk::ImageLayout                       newLayout,
			vk::PipelineStageFlags2               dstStage,
			vk::AccessFlags2                      dstAccess,
			std::vector<vk::ImageMemoryBarrier2>& out
		) {
			const bool readAfterRead = tex.GetCurrentLayout() == newLayout && !IsWrite(tex.GetLastAccess()) &&
				!IsWrite(dstAccess);
			if (readAfterRead) {
				tex.SetLastStageAccess(tex.GetLastStage() | dstStage, tex.GetLastAccess() | dstAccess);
				return;
			}

			const auto& desc = tex.GetDesc();
			out.push_back(
				vk::ImageMemoryBarrier2{
					tex.GetLastStage() ? tex.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe,
					tex.GetLastAccess(),
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
		}

		static void EmitBufferBarrier(
			PhysicalBuffer&                        buf,
			vk::PipelineStageFlags2                dstStage,
			vk::AccessFlags2                       dstAccess,
			std::vector<vk::BufferMemoryBarrier2>& out
		) {
			// No layout for a buffer, so the skip rule's layout-equality half is vacuously true --
			// a read-after-read is just "neither side is a write."
			const bool readAfterRead = !IsWrite(buf.GetLastAccess()) && !IsWrite(dstAccess);
			if (readAfterRead) {
				buf.SetLastStageAccess(buf.GetLastStage() | dstStage, buf.GetLastAccess() | dstAccess);
				return;
			}

			const auto& desc = buf.GetDesc();
			out.push_back(
				vk::BufferMemoryBarrier2{
					buf.GetLastStage() ? buf.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe,
					buf.GetLastAccess(),
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
			std::vector<vk::MemoryBarrier2>& out
		) {
			const bool readAfterRead = !IsWrite(as.GetLastAccess()) && !IsWrite(dstAccess);
			if (readAfterRead) {
				as.SetLastStageAccess(as.GetLastStage() | dstStage, as.GetLastAccess() | dstAccess);
				return;
			}

			out.push_back(
				vk::MemoryBarrier2{
					as.GetLastStage() ? as.GetLastStage() : vk::PipelineStageFlagBits2::eTopOfPipe,
					as.GetLastAccess(),
					dstStage,
					dstAccess,
				}
			);
			as.SetLastStageAccess(dstStage, dstAccess);
		}

		// The producer side of a cross-domain edge (Graph.hpp's SynthesizeBarrier posts these to
		// the producer's own stage). Deriving a layout here would be meaningless -- it would come
		// from the producer's access and the *consumer's* domain, which describes neither side
		// correctly -- and it would double-transition the resource, since the edge's Acquire-
		// phase entry in the consumer's stage already performs the real transition. Worse. one
		// of these can legitimately fire while a resource is still eUndefined: Import<K> nodes
		// run in ExecutionDomain::Host with no realizations of their own (Node.hpp), so a
		// Graphics consumer of an imported key produces a Host->Graphics edge whose release half
		// sits in the Host stage before anything has touched the resource. A per-image barrier
		// there would be newLayout=eUndefined -- exactly Bug 1 this rewrite exists to fix. A
		// single global memory barrier has no layout to get wrong.
		//
		// There is also no real queue-family ownership transfer happening here to "release":
		// VK_QUEUE_FAMILY_IGNORED everywhere, one graphicsQueue in the whole engine. This is
		// future work behind an ExecutionDomain -> queue-family mapping, not attempted here.
		static void
		DispatchRelease(vk::CommandBuffer cmd, PhysicalResourceRegistry& registry, const BarrierBatch& batch) {
			vk::PipelineStageFlags2 srcStage{};
			vk::AccessFlags2        srcAccess{};

			for (const auto& mb : batch.Items()) {
				if (auto tex = registry.GetTexture(mb.resource)) {
					srcStage |= tex->GetLastStage();
					srcAccess |= tex->GetLastAccess();
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

			if (!srcStage && !srcAccess) {
				return;
			}

			vk::MemoryBarrier2 mb2{
				srcStage ? srcStage : vk::PipelineStageFlagBits2::eTopOfPipe,
				srcAccess,
				vk::PipelineStageFlagBits2::eAllCommands,
				vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
			};
			vk::DependencyInfo depInfo{};
			depInfo.setMemoryBarriers(mb2);
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
