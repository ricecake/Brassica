#pragma once

#include <cstdint>
#include <filesystem>

#include "graph/Node.hpp"
#include "model/GltfModel.hpp"
#include "model/GltfTextureManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/MaterialManager.hpp"

namespace brassica::graph {
	class Graph;
	class PhysicalResourceRegistry;
}

namespace brassica {

	// Owns exactly one loaded GLTF model's GPU buffers and textures for Stages 1-2 of the
	// pipeline (static mesh loading + GPU-driven rendering, materials + textures) -- not a
	// general asset manager: one model, loaded once at startup, never mutated at runtime.
	// GltfRenderNode is auto-registered (BRASSICA_REGISTER_NODE) and so has no direct reference
	// to this class; it reaches GetMeshletCount() through ServiceLocator, the same indirection
	// FoliageArchetypeRenderNode uses for IFoliageManager (see its own SetFrameParams()).
	class GltfModelManager {
	public:
		// Loads the model's geometry (GltfModel), uploads its images into real bindless textures
		// (GltfTextureManager) and registers its materials into the *shared* MaterialManager --
		// not a parallel GLTF-only material buffer, per the explicit decision to actually use the
		// existing material system rather than work around it. Needs a live device (registry) and
		// a MaterialManager instance that's already Initialize()'d -- call after both exist.
		// Returns false (and leaves any previously loaded model buffers in place) on parse
		// failure -- see GltfModel::LoadFromFile for the real error logging.
		bool LoadFromFile(
			const std::filesystem::path&     path,
			graph::PhysicalResourceRegistry& registry,
			MaterialManager&                 materialManager
		);

		[[nodiscard]] std::uint32_t GetMeshletCount() const {
			return static_cast<std::uint32_t>(m_model.Meshlets().size());
		}

		// Stage 4 (GPU skinning): re-evaluates the current animation's pose at `time` (seconds,
		// any value -- wrapped to [0, duration) internally against the active clip's own
		// duration, so the caller never needs to track looping itself) and re-uploads the
		// resulting joint matrices. A no-op-but-safe single-identity-matrix upload when the
        // model has no skin at all (JointCount() == 0), so the buffer/address is always valid
        // even though gltf_render.mesh never actually reads it for an unskinned vertex (see its
        // own weight-sum check). Call every frame, same as RegisterBufferNodes below -- unlike
        // the other four GLTF buffers, this one's contents genuinely change every frame.
		void UpdatePose(float time);

		// Mirrors MaterialManager::RegisterBufferNode's exact shape and call sites (Engine.cpp
		// rebuilds graph::Graph fresh every frame, so every PredefinedBufferNode-owning manager
		// needs its own RegisterRef call repeated there, not just once at startup).
		void RegisterBufferNodes(graph::Graph& frameGraph);

	private:
		GltfModel           m_model;
		GltfTextureManager  m_textureManager;

		// Which of the model's animation clips plays -- a plain member field with no selection
		// UI/manager yet, same placeholder-scope choice as GltfRenderNode's own modelWorldPosition/
		// modelScale. 0 is simply "the first clip in the asset," not a meaningful default pick.
		std::uint32_t m_currentAnimationIndex{0};

		graph::PredefinedBufferNode<GltfVertexBuffer, GltfVertex, SubPhase::Prepare>            m_vertexBufNode;
		graph::PredefinedBufferNode<GltfMeshletVertexBuffer, std::uint32_t, SubPhase::Prepare>   m_meshletVertexBufNode;
		graph::PredefinedBufferNode<GltfMeshletTriangleBuffer, std::uint32_t, SubPhase::Prepare> m_meshletTriangleBufNode;
		graph::PredefinedBufferNode<GltfMeshletBuffer, GltfMeshlet, SubPhase::Prepare>           m_meshletBufNode;
		graph::PredefinedBufferNode<GltfJointMatrixBuffer, glm::mat4, SubPhase::Prepare>         m_jointMatrixBufNode;
	};

} // namespace brassica
