#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vk_mem_alloc.h"
#include "vulkan/vulkan.hpp"

#include "graph/ResourceKey.hpp"

namespace brassica::graph {
	class PhysicalResourceRegistry;
}

namespace brassica {

	// One pre-registered dumpable texture: a human label plus the resource key's runtime
	// identity, captured once via graph::IdOf<K>() at registration time. RegisterDumpable<K>()
	// is the only place a concrete key type needs naming -- adding the next dumpable texture
	// is exactly one call, no dynamic/reflection machinery.
	struct TextureDumpEntry {
		std::string       label;
		graph::ResourceId id;
	};

	// Debug-only, click-a-button GPU->CPU texture dump to a PNG on disk. Deliberately
	// synchronous (blocks briefly on queue.waitIdle(), same accepted cost
	// TerrainMapExporter::ExportGPU already pays for its own one-shot export) rather than a
	// frame-graph node -- this is a rare, user-triggered inspection action, not a per-frame
	// system. See TerrainReadbackNode.hpp for why a *continuous* readback needs the heavier
	// async/fence-tracked machinery this deliberately skips.
	//
	// Only reads array layer 0 / mip 0 / (for a 3D image) depth slice 0 of whatever resource is
	// requested -- a multi-layer LOD array or a volume texture only ever gets that one slice.
	// There's no single correct "the" slice to pick for those in general, so this doesn't
	// pretend to -- Dump() logs which slice it took.
	class TextureDumpService {
	public:
		void Init(
			vk::Device device, vk::Queue queue, std::uint32_t queueFamily, VmaAllocator allocator,
			graph::PhysicalResourceRegistry& registry
		);

		template <typename K>
		void RegisterDumpable(std::string label) {
			m_entries.push_back(TextureDumpEntry{std::move(label), graph::IdOf<K>()});
		}

		[[nodiscard]] const std::vector<TextureDumpEntry>& Entries() const { return m_entries; }

		// Synchronous: copies the texture to a staging buffer, converts it to an 8-bit PNG
		// (each channel independently min/max-normalized across the image, so arbitrary
		// simulation-value ranges still produce a readable image instead of a solid clamped
		// block), and writes it to outputDir/<sanitized label>_<timestamp>.png (directory
		// created if missing). Returns false (with a logged reason) if the resource isn't
		// currently provisioned, has no defined contents yet, uses a format this doesn't
		// support, or the GPU/file-write step fails.
		bool Dump(const TextureDumpEntry& entry, const std::string& outputDir);

	private:
		vk::Device                       m_device{nullptr};
		vk::Queue                        m_queue{nullptr};
		std::uint32_t                    m_queueFamily{0};
		VmaAllocator                     m_allocator{nullptr};
		graph::PhysicalResourceRegistry* m_registry{nullptr};
		std::vector<TextureDumpEntry>    m_entries;
	};

} // namespace brassica
