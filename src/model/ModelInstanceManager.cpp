#include "model/ModelInstanceManager.hpp"

#include "graph/Graph.hpp"

namespace brassica {

	void ModelInstanceManager::RegisterBufferNodes(graph::Graph& frameGraph) {
		m_lastFlushCount = static_cast<std::uint32_t>(m_dirtySlots.size());
		if (m_lastFlushCount > 0) {
			std::vector<ModelInstanceWrite> writes;
			writes.reserve(m_dirtySlots.size());
			for (std::uint32_t slot : m_dirtySlots) {
				writes.push_back(ModelInstanceWrite{.slot = slot, .record = m_records[slot]});
			}
			// Explicit, fixed-capacity desc (not the size-derived default) -- see the constructor's
			// own comment on why this must stay the exact same desc every call, including the
			// very first one. ModelInstanceScatterNode reads this buffer via a bindless
			// storage-buffer index (ctx.BufferIndex<K>(), binding 6), not a device address.
			m_writeBufNode.SetData(writes, WriteBufferDesc(m_capacity));
			m_dirtySlots.clear();
		}
		frameGraph.RegisterRef(m_writeBufNode);
	}

} // namespace brassica
