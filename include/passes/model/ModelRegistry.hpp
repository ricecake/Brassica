#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "graph/Node.hpp"
#include "passes/model/ModelAsset.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/model/resources.hpp"

namespace brassica::graph {
	class Graph;
	class ResourceCatalog;
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
		// model's index, for ModelInstance::modelIndex. name is a human-readable label for debug
		// UI (ModelInstanceWidget's model picker) -- purely cosmetic, never read by any shader or
		// GPU-side code.
		std::uint32_t RegisterAsset(const ModelAsset& asset, std::string name = {});

		// Procedural plant generation v2's ad-hoc designer: overwrites one already-registered
		// model's data in place (re-concatenating the shared buffers, same "re-upload everything"
		// shape as RegisterAsset) instead of appending a new model index -- RegisterAsset is
		// append-only and was never meant to be hammered by a Generate/Regenerate button, which
        // would otherwise leak one more permanently-resident model every press. index must
		// already be a real, previously-registered model (from RegisterAsset or an earlier
		// ReplaceAsset); no-op if out of range. Caller contract: despawn any live instance of
		// this model index *before* calling this, then spawn fresh after -- a live instance
		// still pointing at this index mid-call would read a stale jointCount/meshletCount for
		// one frame otherwise.
		void ReplaceAsset(std::uint32_t index, const ModelAsset& asset);

		[[nodiscard]] std::uint32_t ModelCount() const { return static_cast<std::uint32_t>(m_descs.size()); }

		[[nodiscard]] const std::string& GetName(std::uint32_t modelIndex) const {
			static const std::string kEmpty;
			return modelIndex < m_names.size() ? m_names[modelIndex] : kEmpty;
		}

		[[nodiscard]] std::uint32_t GetMeshletCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].meshletCount : 0u;
		}

		[[nodiscard]] std::uint32_t GetJointCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].jointCount : 0u;
		}

		[[nodiscard]] std::uint32_t GetClipCount(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].clipCount : 0u;
		}

		[[nodiscard]] std::uint32_t GetVertexBase(std::uint32_t modelIndex) const {
			return modelIndex < m_descs.size() ? m_descs[modelIndex].vertexBase : 0u;
		}

		// hasNext ? next model's own vertexBase : total vertex count -- the same derivation
		// ReplaceAsset already relies on internally (ModelRegistry.cpp) for every shared buffer,
		// since UploadAll() always re-concatenates every model contiguously in registration order.
		[[nodiscard]] std::uint32_t GetVertexCount(std::uint32_t modelIndex) const {
			if (modelIndex >= m_descs.size()) {
				return 0u;
			}
			std::uint32_t base = m_descs[modelIndex].vertexBase;
			std::uint32_t end = (modelIndex + 1 < m_descs.size()) ? m_descs[modelIndex + 1].vertexBase
																   : static_cast<std::uint32_t>(m_vertices.size());
			return end - base;
		}

		// Flat, model-local (0-based) triangle index list for the decor acceleration structure's
		// BLAS geometry -- see ModelFlatIndexRange's own comment for how this is derived.
		[[nodiscard]] std::span<const std::uint32_t> GetFlatTriangleIndices() const { return m_flatTriangleIndices; }

		// For DecorAccelerationStructure's own vertex-position buffer copy -- deliberately NOT
		// read from the shared catalog GltfVertexBuffer via device address: that buffer is
		// MinimalDevice-reachable generic infrastructure (model registration, wind bounds,
		// procedural generation tests all touch it on a device with no AS support at all -- see
		// tests/MinimalDevice.hpp's own "do not add... acceleration structure... requirements
		// here" comment), so tagging it with AS-build usage broke every one of those tests. A
		// private copy, owned entirely by DecorAccelerationStructure, keeps that requirement
		// scoped to code that actually needs it.
		[[nodiscard]] std::span<const GltfVertex> GetVertices() const { return m_vertices; }

		[[nodiscard]] ModelFlatIndexRange GetFlatIndexRange(std::uint32_t modelIndex) const {
			return modelIndex < m_flatIndexRanges.size() ? m_flatIndexRanges[modelIndex] : ModelFlatIndexRange{};
		}

		// Bumped once per UploadAll() call (every RegisterAsset/ReplaceAsset) -- the one signal
		// DecorAccelerationStructure needs to know its per-template BLASes are stale, without
		// inventing a second change-detection mechanism alongside the registry's own existing
		// "re-upload everything" trigger.
		[[nodiscard]] std::uint64_t GetVersion() const { return m_version; }

		// Total meshlets across every registered model -- ModelRenderNode's own Active() gate
		// ("is there anything loaded at all") reads this, same role GltfModelManager's
		// GetMeshletCount() played for the single-model Stage 1 node.
		[[nodiscard]] std::uint32_t GetTotalMeshletCount() const { return static_cast<std::uint32_t>(m_meshlets.size()); }

		// Same per-frame-rebuild shape as every other PredefinedBufferNode-owning manager --
		// Engine.cpp rebuilds graph::Graph fresh every frame, so this needs calling every frame.
		// Publishes every buffer's current byteSize onto its own catalog instance first (same
		// pattern as MaterialManager::RegisterBufferNode) -- post-S2b, that's the only way a
		// genuine-runtime-config byte size reaches a key that isn't a type-level constant.
		void RegisterBufferNodes(graph::Graph& frameGraph, graph::ResourceCatalog& catalog);

	private:
		// Re-stages every shared buffer + the desc buffer from the current member vectors --
		// shared by RegisterAsset (append) and ReplaceAsset (in-place overwrite), both of which
		// need to re-upload everything regardless of which model actually changed.
		void UploadAll();

		// Re-derives m_flatTriangleIndices/m_flatIndexRanges from the current m_descs/m_meshlets/
		// m_meshletVertices/m_meshletTrianglesPacked, walking exactly the same meshlet-local-byte
		// -> meshlet-vertex-remap -> model-local-index double indirection model_render.mesh
		// resolves per-vertex at draw time (shaders/model_render.mesh), so the two never drift.
		// Called from UploadAll() -- same "re-upload everything on any registry change" trigger
		// as every other derived vector here, no separate change-detection mechanism.
		void RebuildFlatTriangleIndices();

		std::vector<GltfVertex>     m_vertices;
		std::vector<std::uint32_t>  m_meshletVertices;
		std::vector<std::uint32_t>  m_meshletTrianglesPacked;
		std::vector<GltfMeshlet>    m_meshlets;
		std::vector<glm::mat4>          m_palette;
		std::vector<ModelClipEntry>     m_clipTable;
		std::vector<ModelDescEntry>     m_descs;
		std::vector<ModelSkeletonJoint> m_skeleton;
		std::vector<std::string>        m_names;

		std::vector<std::uint32_t>      m_flatTriangleIndices;
		std::vector<ModelFlatIndexRange> m_flatIndexRanges;
		std::uint64_t                    m_version{0};

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
