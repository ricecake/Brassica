#include "passes/model/ModelInstanceManager.hpp"

#include "graph/Graph.hpp"
#include "graph/ResourceCatalog.hpp"

namespace brassica {

	void ModelInstanceManager::RegisterBufferNodes(graph::Graph& frameGraph, graph::ResourceCatalog& catalog) {
		// Explicit, fixed-capacity byteSize (not derived from this frame's write count) -- see
		// the constructor's own comment on why this must stay the exact same value every call,
		// including the very first one.
		catalog.Get<ModelInstanceWriteBuffer>().byteSize =
			static_cast<std::uint64_t>(m_capacity) * sizeof(ModelInstanceWrite);

		m_lastFlushCount = static_cast<std::uint32_t>(m_dirtySlots.size());
		if (m_lastFlushCount > 0) {
			std::vector<ModelInstanceWrite> writes;
			writes.reserve(m_dirtySlots.size());
			for (std::uint32_t slot : m_dirtySlots) {
				writes.push_back(ModelInstanceWrite{.slot = slot, .record = m_records[slot]});
			}
			// ModelInstanceScatterNode reads this buffer via a bindless storage-buffer index
			// (ctx.BufferIndex<K>(), binding 6), not a device address.
			m_writeBufNode.SetData(writes);
			m_dirtySlots.clear();
		}
		frameGraph.RegisterRef(m_writeBufNode);
	}

} // namespace brassica
