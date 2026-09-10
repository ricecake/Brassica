#pragma once
#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "graph/ResourceKey.hpp"

// This is the entire Vulkan seam for the graph skeleton. Nothing in this file, or anything
// it includes, may depend on vulkan/vulkan.hpp, VMA, or external/FrameGraph. When a real
// backend lands, this is the only header whose stubs get replaced -- everything else in
// include/graph/ only ever sees these types.

namespace brassica::graph {

	enum class ExecutionDomain : std::uint8_t { Graphics, Compute, Transfer, Host };

	enum class AccessKind : std::uint8_t { Read, Write, ReadWrite };

	// Stand-in for a backend command buffer. Deliberately opaque at this layer.
	struct CommandBuffer {
		void* vkCmd = nullptr;
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

	// The concrete realization of a resource key for this frame, factoring in things like
	// screen resolution -- the "more concrete type" a node produces at Setup time, as opposed
	// to the abstract key it declares at the class level. formatCode/usageMask are opaque
	// here on purpose: this layer never interprets them, it only carries them through (and,
	// later, compares/hashes them for aliasing) for a backend to map onto real enums.
	struct ResourceDesc {
		enum class Kind : std::uint8_t { Image2D, Image3D, Buffer };

		Kind          kind = Kind::Image2D;
		std::uint32_t width = 0;
		std::uint32_t height = 0;
		std::uint32_t depth = 1;
		std::uint32_t mips = 1;
		std::uint32_t layers = 1;
		std::uint32_t formatCode = 0;
		std::uint32_t usageMask = 0;
		std::uint64_t byteSize = 0;
	};

	struct ResourceRealization {
		ResourceId   key = nullptr;
		AccessKind   access = AccessKind::Read;
		ResourceDesc desc{};
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
