#pragma once

#include "vulkan/vulkan.h"

namespace brassica {

	struct FrameGraphTexture {
		struct Desc {
			VkExtent2D extent{0, 0};
			VkFormat   format{VK_FORMAT_UNDEFINED};
		};

		VkImage     image{VK_NULL_HANDLE};
		VkImageView imageView{VK_NULL_HANDLE};

		FrameGraphTexture() = default;
		FrameGraphTexture(VkImage img, VkImageView view) : image(img), imageView(view) {}
		FrameGraphTexture(FrameGraphTexture&&) noexcept = default;
		FrameGraphTexture& operator=(FrameGraphTexture&&) noexcept = default;

		void create(const Desc& desc, void* context) {}
		void destroy(const Desc& desc, void* context) {}
	};

} // namespace brassica
