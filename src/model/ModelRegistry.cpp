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

		return modelIndex;
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
