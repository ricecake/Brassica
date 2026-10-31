#include "model/GltfTextureManager.hpp"

namespace brassica {

	std::vector<std::uint32_t> GltfTextureManager::UploadAll(
		graph::PhysicalResourceRegistry& registry,
		const std::vector<GltfImage>&    images
	) {
		std::vector<std::uint32_t> bindlessIndices;
		bindlessIndices.reserve(images.size());

		for (const GltfImage& image : images) {
			graph::ResourceDesc desc{
				.kind = graph::ResourceDesc::Kind::Image2D,
				.width = image.width,
				.height = image.height,
				.formatCode =
					static_cast<std::uint32_t>(image.isSrgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm),
				.usageMask =
					static_cast<std::uint32_t>(vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst),
				// Lives for the engine's lifetime, same as GltfModelManager's own buffers -- not
				// provisioned through the per-frame graph/alias pool at all (CreateAndUploadTexture2D
				// bypasses Provision entirely), so `persistent` has no actual effect here, but it
				// documents the real lifetime intent rather than defaulting to false by omission.
				.persistent = true,
			};

			auto tex = registry.CreateAndUploadTexture2D(image.pixelsRGBA8.data(), image.pixelsRGBA8.size(), desc);
			m_textures.push_back(tex);
			bindlessIndices.push_back(tex->GetSampledBindlessIndex());
		}

		return bindlessIndices;
	}

} // namespace brassica
