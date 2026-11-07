#pragma once

#include <filesystem>

#include "passes/model/GltfModel.hpp"
#include "passes/model/GltfTextureManager.hpp"
#include "passes/material/MaterialManager.hpp"

namespace brassica::graph {
	class PhysicalResourceRegistry;
}

namespace brassica {

	struct ModelAsset;

	// Stage 2: the GLTF/GLB producer -- parses a single asset (GltfModel), uploads its images
	// into real bindless textures (GltfTextureManager), registers its materials into the
	// *shared* MaterialManager, and bakes its clips, then hands all of it over as a producer-
	// agnostic ModelAsset for ModelRegistry::RegisterAsset() to concatenate. No buffer ownership
	// here anymore (that moved to ModelRegistry, which has to own shared buffers across every
	// producer, not just this one) -- this class is purely "glTF file -> ModelAsset," the same
	// role a procedural asset builder plays for its own producer.
	class GltfModelManager {
	public:
		// Returns false (and leaves outAsset untouched) on parse failure -- see
		// GltfModel::LoadFromFile for the real error logging. Needs a live device (registry) and
		// a MaterialManager instance that's already Initialize()'d -- call after both exist.
		bool LoadFromFile(
			const std::filesystem::path&     path,
			graph::PhysicalResourceRegistry& registry,
			MaterialManager&                 materialManager,
			ModelAsset&                      outAsset
		);

	private:
		GltfModel          m_model;
		GltfTextureManager m_textureManager;
	};

} // namespace brassica
