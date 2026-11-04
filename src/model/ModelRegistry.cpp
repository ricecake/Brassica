#include "model/ModelRegistry.hpp"

#include "graph/Graph.hpp"

namespace brassica {

	std::uint32_t ModelRegistry::RegisterAsset(const ModelAsset& asset) {
		ModelDescEntry desc{};
		desc.restBoundsCenter = asset.restBoundsCenter;
		desc.restBoundsRadius = asset.restBoundsRadius;
		desc.vertexBase = static_cast<std::uint32_t>(m_vertices.size());
		desc.meshletVertexBase = static_cast<std::uint32_t>(m_meshletVertices.size());
		desc.meshletTriangleWordBase = static_cast<std::uint32_t>(m_meshletTrianglesPacked.size());
		desc.meshletBase = static_cast<std::uint32_t>(m_meshlets.size());
		desc.meshletCount = static_cast<std::uint32_t>(asset.meshlets.size());
		desc.jointCount = asset.jointCount;
		desc.clipTableOffset = static_cast<std::uint32_t>(m_clipTable.size());
		desc.clipCount = static_cast<std::uint32_t>(asset.clipTable.size());
		desc.restPaletteBase = static_cast<std::uint32_t>(m_palette.size());
		// kModelNoSkeletonSentinel, not a real base, for a producer that never fills
		// ModelAsset::skeleton (e.g. GltfModelManager's cow) -- see ModelDescEntry's own comment.
		desc.skeletonBase = asset.skeleton.empty()
			? kModelNoSkeletonSentinel
			: static_cast<std::uint32_t>(m_skeleton.size());

		std::uint32_t modelIndex = static_cast<std::uint32_t>(m_descs.size());
		m_descs.push_back(desc);

		m_vertices.insert(m_vertices.end(), asset.vertices.begin(), asset.vertices.end());
		m_meshletVertices.insert(m_meshletVertices.end(), asset.meshletVertices.begin(), asset.meshletVertices.end());
		m_meshletTrianglesPacked.insert(
			m_meshletTrianglesPacked.end(),
			asset.meshletTrianglesPacked.begin(),
			asset.meshletTrianglesPacked.end()
		);
		m_meshlets.insert(m_meshlets.end(), asset.meshlets.begin(), asset.meshlets.end());
		m_palette.insert(m_palette.end(), asset.palette.begin(), asset.palette.end());
		m_clipTable.insert(m_clipTable.end(), asset.clipTable.begin(), asset.clipTable.end());
		m_skeleton.insert(m_skeleton.end(), asset.skeleton.begin(), asset.skeleton.end());

		UploadAll();
		return modelIndex;
	}

	namespace {
		// Erases the old [base, base+oldCount) range and inserts newData at that same starting
		// position -- the one mechanic ReplaceAsset repeats per shared buffer.
		template <typename T>
		void ReplaceRange(std::vector<T>& data, std::uint32_t base, std::uint32_t oldCount, const std::vector<T>& newData) {
			data.erase(data.begin() + base, data.begin() + base + oldCount);
			data.insert(data.begin() + base, newData.begin(), newData.end());
		}
	} // namespace

	void ModelRegistry::ReplaceAsset(std::uint32_t index, const ModelAsset& asset) {
		if (index >= m_descs.size()) {
			return;
		}

		ModelDescEntry oldDesc = m_descs[index];
		bool           hasNext = (index + 1 < m_descs.size());

		std::uint32_t oldVertexCount = hasNext ? m_descs[index + 1].vertexBase - oldDesc.vertexBase
												: static_cast<std::uint32_t>(m_vertices.size()) - oldDesc.vertexBase;
		std::uint32_t oldMeshletVertexCount = hasNext
			? m_descs[index + 1].meshletVertexBase - oldDesc.meshletVertexBase
			: static_cast<std::uint32_t>(m_meshletVertices.size()) - oldDesc.meshletVertexBase;
		std::uint32_t oldTriangleWordCount = hasNext
			? m_descs[index + 1].meshletTriangleWordBase - oldDesc.meshletTriangleWordBase
			: static_cast<std::uint32_t>(m_meshletTrianglesPacked.size()) - oldDesc.meshletTriangleWordBase;
		std::uint32_t oldMeshletCount = oldDesc.meshletCount;
		std::uint32_t oldClipCount = oldDesc.clipCount;
		std::uint32_t oldPaletteCount = hasNext ? m_descs[index + 1].restPaletteBase - oldDesc.restPaletteBase
												 : static_cast<std::uint32_t>(m_palette.size()) - oldDesc.restPaletteBase;
		std::uint32_t oldSkeletonCount = (oldDesc.skeletonBase == kModelNoSkeletonSentinel) ? 0u : oldDesc.jointCount;

		ReplaceRange(m_vertices, oldDesc.vertexBase, oldVertexCount, asset.vertices);
		ReplaceRange(m_meshletVertices, oldDesc.meshletVertexBase, oldMeshletVertexCount, asset.meshletVertices);
		ReplaceRange(m_meshletTrianglesPacked, oldDesc.meshletTriangleWordBase, oldTriangleWordCount, asset.meshletTrianglesPacked);
		ReplaceRange(m_meshlets, oldDesc.meshletBase, oldMeshletCount, asset.meshlets);
		ReplaceRange(m_palette, oldDesc.restPaletteBase, oldPaletteCount, asset.palette);
		ReplaceRange(m_clipTable, oldDesc.clipTableOffset, oldClipCount, asset.clipTable);
		std::uint32_t newSkeletonBase = oldDesc.skeletonBase;
		if (oldDesc.skeletonBase != kModelNoSkeletonSentinel) {
			ReplaceRange(m_skeleton, oldDesc.skeletonBase, oldSkeletonCount, asset.skeleton);
		} else if (!asset.skeleton.empty()) {
			// Was jointless, now isn't -- there's no old base to reuse, so insert wherever the
			// next later model's own real skeleton starts (preserving registration order among
			// skeleton-owning models), or at the end if no later model has one.
			newSkeletonBase = static_cast<std::uint32_t>(m_skeleton.size());
			for (std::size_t j = index + 1; j < m_descs.size(); ++j) {
				if (m_descs[j].skeletonBase != kModelNoSkeletonSentinel) {
					newSkeletonBase = m_descs[j].skeletonBase;
					break;
				}
			}
			m_skeleton.insert(m_skeleton.begin() + newSkeletonBase, asset.skeleton.begin(), asset.skeleton.end());
		}

		std::int64_t vertexDelta = static_cast<std::int64_t>(asset.vertices.size()) - oldVertexCount;
		std::int64_t meshletVertexDelta = static_cast<std::int64_t>(asset.meshletVertices.size()) - oldMeshletVertexCount;
		std::int64_t triangleWordDelta = static_cast<std::int64_t>(asset.meshletTrianglesPacked.size()) - oldTriangleWordCount;
		std::int64_t meshletDelta = static_cast<std::int64_t>(asset.meshlets.size()) - oldMeshletCount;
		std::int64_t paletteDelta = static_cast<std::int64_t>(asset.palette.size()) - oldPaletteCount;
		std::int64_t clipDelta = static_cast<std::int64_t>(asset.clipTable.size()) - oldClipCount;
		std::int64_t skeletonDelta = static_cast<std::int64_t>(asset.skeleton.size()) - oldSkeletonCount;

		for (std::size_t j = index + 1; j < m_descs.size(); ++j) {
			m_descs[j].vertexBase = static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].vertexBase) + vertexDelta);
			m_descs[j].meshletVertexBase =
				static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].meshletVertexBase) + meshletVertexDelta);
			m_descs[j].meshletTriangleWordBase = static_cast<std::uint32_t>(
				static_cast<std::int64_t>(m_descs[j].meshletTriangleWordBase) + triangleWordDelta
			);
			m_descs[j].meshletBase = static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].meshletBase) + meshletDelta);
			m_descs[j].restPaletteBase =
				static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].restPaletteBase) + paletteDelta);
			m_descs[j].clipTableOffset =
				static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].clipTableOffset) + clipDelta);
			if (m_descs[j].skeletonBase != kModelNoSkeletonSentinel) {
				m_descs[j].skeletonBase =
					static_cast<std::uint32_t>(static_cast<std::int64_t>(m_descs[j].skeletonBase) + skeletonDelta);
			}
		}

		ModelDescEntry newDesc{};
		newDesc.restBoundsCenter = asset.restBoundsCenter;
		newDesc.restBoundsRadius = asset.restBoundsRadius;
		newDesc.vertexBase = oldDesc.vertexBase;
		newDesc.meshletVertexBase = oldDesc.meshletVertexBase;
		newDesc.meshletTriangleWordBase = oldDesc.meshletTriangleWordBase;
		newDesc.meshletBase = oldDesc.meshletBase;
		newDesc.meshletCount = static_cast<std::uint32_t>(asset.meshlets.size());
		newDesc.jointCount = asset.jointCount;
		newDesc.clipTableOffset = oldDesc.clipTableOffset;
		newDesc.clipCount = static_cast<std::uint32_t>(asset.clipTable.size());
		newDesc.restPaletteBase = oldDesc.restPaletteBase;
		newDesc.skeletonBase = asset.skeleton.empty() ? kModelNoSkeletonSentinel : newSkeletonBase;
		m_descs[index] = newDesc;

		UploadAll();
	}

	void ModelRegistry::UploadAll() {
		// Staged (not device-address-flagged): every buffer here is read via a bindless
		// storage-buffer index (ctx.BufferIndex<K>(), binding 6), not buffer_reference.
		m_vertexBufNode.SetData(m_vertices, graph::StagedStorageBufferDesc(m_vertices.size() * sizeof(GltfVertex)));
		m_meshletVertexBufNode.SetData(
			m_meshletVertices,
			graph::StagedStorageBufferDesc(m_meshletVertices.size() * sizeof(std::uint32_t))
		);
		m_meshletTriangleBufNode.SetData(
			m_meshletTrianglesPacked,
			graph::StagedStorageBufferDesc(m_meshletTrianglesPacked.size() * sizeof(std::uint32_t))
		);
		m_meshletBufNode.SetData(m_meshlets, graph::StagedStorageBufferDesc(m_meshlets.size() * sizeof(GltfMeshlet)));
		m_paletteBufNode.SetData(m_palette, graph::StagedStorageBufferDesc(m_palette.size() * sizeof(glm::mat4)));
		m_clipTableBufNode.SetData(
			m_clipTable,
			graph::StagedStorageBufferDesc(m_clipTable.size() * sizeof(ModelClipEntry))
		);
		m_descBufNode.SetData(m_descs, graph::StagedStorageBufferDesc(m_descs.size() * sizeof(ModelDescEntry)));
		m_skeletonBufNode.SetData(
			m_skeleton,
			graph::StagedStorageBufferDesc(m_skeleton.size() * sizeof(ModelSkeletonJoint))
		);
	}

	void ModelRegistry::RegisterBufferNodes(graph::Graph& frameGraph) {
		frameGraph.RegisterRef(m_vertexBufNode);
		frameGraph.RegisterRef(m_meshletVertexBufNode);
		frameGraph.RegisterRef(m_meshletTriangleBufNode);
		frameGraph.RegisterRef(m_meshletBufNode);
		frameGraph.RegisterRef(m_paletteBufNode);
		frameGraph.RegisterRef(m_clipTableBufNode);
		frameGraph.RegisterRef(m_descBufNode);
		frameGraph.RegisterRef(m_skeletonBufNode);
	}

} // namespace brassica
