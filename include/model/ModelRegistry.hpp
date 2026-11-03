#pragma once

#include <cstdint>
#include <vector>

#include "graph/Node.hpp"
#include "model/ModelAsset.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"

namespace brassica::graph {
	class Graph;
}

namespace brassica {

	// Stage 2: owns every registered model's shared GPU buffers. The engine's resource-key
	// system is one-physical-buffer-per-C++-type (no per-instance-indexed keys -- VersionedKey
	// was removed in an earlier graph rework), so N models can't each own their own vertex/
	// meshlet/palette/clip-table buffer the way Stage 1's single-model GltfModelManager did.
	// Instead this concatenates every RegisterAsset() call's data into one shared buffer per
	// kind, and ModelDescBuffer records where each model's own chunk starts -- see
	// ModelDescEntry's own comment (model/ModelTypes.hpp) for why the shader adds that base at
	// the lookup point instead of this class rewriting any of the data itself.
	//
	// Producer-agnostic by design: GltfModelManager (a real glTF/GLB file) is one producer,
	// a hand-built procedural asset is another -- both just build a ModelAsset and call
	// RegisterAsset(), with no further coupling to this class at all.
	class ModelRegistry {
	public:
		// Concatenates asset's data into the shared buffers and re-uploads everything
		// (RegisterAsset is a startup-time call, a handful of times at most -- not a hot path,
		// so re-uploading the whole growing buffer set each time stays simple). Returns the new
		// model's index, for ModelInstance::modelIndex.
		std::uint32_t RegisterAsset(const ModelAsset& asset);

		[[nodiscard]] std::uint32_t ModelCount() const { return static_cast<std::uint32_t>(m_descs.size()); }

		[[nodiscard]] std::uint32_t GetMeshletCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].meshletCount : 0u;
		}

		[[nodiscard]] std::uint32_t GetJointCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].jointCount : 0u;
		}

		[[nodiscard]] std::uint32_t GetClipCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].clipCount : 0u;
		}

		// Total meshlets across every registered model -- ModelRenderNode's own Active() gate
		// ("is there anything loaded at all") reads this, same role GltfModelManager's
		// GetMeshletCount() played for the single-model Stage 1 node.
		[[nodiscard]] std::uint32_t GetTotalMeshletCount() const { return static_cast<std::uint32_t>(m_meshlets.size()); }

		// Same per-frame-rebuild shape as every other PredefinedBufferNode-owning manager --
		// Engine.cpp rebuilds graph::Graph fresh every frame, so this needs calling every frame.
		void RegisterBufferNodes(graph::Graph& frameGraph);

	private:
		std::vector<GltfVertex>     m_vertices;
		std::vector<std::uint32_t>  m_meshletVertices;
		std::vector<std::uint32_t>  m_meshletTrianglesPacked;
		std::vector<GltfMeshlet>    m_meshlets;
		std::vector<glm::mat4>          m_palette;
		std::vector<ModelClipEntry>     m_clipTable;
		std::vector<ModelDescEntry>     m_descs;
		std::vector<ModelSkeletonJoint> m_skeleton;

		graph::PredefinedBufferNode<GltfVertexBuffer, GltfVertex, SubPhase::Prepare>            m_vertexBufNode;
		graph::PredefinedBufferNode<GltfMeshletVertexBuffer, std::uint32_t, SubPhase::Prepare>   m_meshletVertexBufNode;
		graph::PredefinedBufferNode<GltfMeshletTriangleBuffer, std::uint32_t, SubPhase::Prepare> m_meshletTriangleBufNode;
		graph::PredefinedBufferNode<GltfMeshletBuffer, GltfMeshlet, SubPhase::Prepare>           m_meshletBufNode;
		graph::PredefinedBufferNode<ModelClipPaletteBuffer, glm::mat4, SubPhase::Prepare>        m_paletteBufNode;
		graph::PredefinedBufferNode<ModelClipTableBuffer, ModelClipEntry, SubPhase::Prepare>     m_clipTableBufNode;
		graph::PredefinedBufferNode<ModelDescBuffer, ModelDescEntry, SubPhase::Prepare>          m_descBufNode;
		graph::PredefinedBufferNode<ModelSkeletonBuffer, ModelSkeletonJoint, SubPhase::Prepare>  m_skeletonBufNode;
	};

} // namespace brassica
