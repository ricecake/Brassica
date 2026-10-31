#pragma once

#include <memory>
#include <vector>

#include "graph/PhysicalRegistry.hpp"
#include "model/GltfModel.hpp"

namespace brassica {

	// Uploads a GltfModel's already-decoded GltfImage buffers (stb_image decode already happened
	// CPU-side in GltfModel -- see its own comment on why) into real bindless-indexed GPU
	// textures. Each call creates brand-new PhysicalTexture objects via
	// PhysicalResourceRegistry::CreateAndUploadTexture2D and keeps them alive for the engine's
	// lifetime (no unload path -- same one-model-loaded-once scope as GltfModelManager).
	class GltfTextureManager {
	public:
		// Returns one bindless sampled-image index per entry in `images`, same order. Every
		// decoded image is expected to upload successfully -- GltfModel only ever appends images
		// it already decoded without error, so there's no per-entry failure sentinel here.
		std::vector<std::uint32_t> UploadAll(graph::PhysicalResourceRegistry& registry, const std::vector<GltfImage>& images);

	private:
		std::vector<std::shared_ptr<graph::PhysicalTexture>> m_textures;
	};

} // namespace brassica
