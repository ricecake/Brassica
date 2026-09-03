#pragma once

#include "vulkan/vulkan.hpp"

namespace brassica {

	struct FrameGraphTexture {
		struct Desc {
			vk::Extent2D extent{0, 0};
			vk::Format   format{vk::Format::eUndefined};
		};

		vk::Image     image{nullptr};
		vk::ImageView imageView{nullptr};

		FrameGraphTexture() = default;

		FrameGraphTexture(vk::Image img, vk::ImageView view): image(img), imageView(view) {}

		FrameGraphTexture(FrameGraphTexture&&) noexcept = default;
		FrameGraphTexture& operator=(FrameGraphTexture&&) noexcept = default;

		void create(const Desc& desc, void* context) {}

		void destroy(const Desc& desc, void* context) {}
	};

} // namespace brassica
