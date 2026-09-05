#include "passes/GradientPass.hpp"

#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	GradientPass::GradientPass(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher)
		: RenderPass("GradientPass", dev, colorFmt) {
		InitPipeline(dev, colorFmt, watcher);
	}

	GradientPass::~GradientPass() {
		if (lastAllocator != VK_NULL_HANDLE) {
			DestroyTextureResource(lastAllocator);
		}
	}

	void GradientPass::DestroyTextureResource(VmaAllocator allocator) {
		if (bgImageView) {
			device.destroyImageView(bgImageView);
			bgImageView = nullptr;
		}
		if (bgImage && bgAllocation) {
			vmaDestroyImage(allocator, bgImage, bgAllocation);
			bgImage = nullptr;
			bgAllocation = VK_NULL_HANDLE;
		}
	}

	void GradientPass::InitPipeline(vk::Device dev, vk::Format colorFmt, ShaderWatcher* watcher) {
		if (!vertShader.CompileVertexFromFile(dev, "shaders/gradient.vert")) {
			spdlog::error("Failed to compile gradient.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/gradient.frag")) {
			spdlog::error("Failed to compile gradient.frag shader file");
		}

		SetShaders(&vertShader, &fragShader);
		InitRenderPipeline(colorFmt, vk::Format::eUndefined, {}, {}, watcher, false, false, vk::CompareOp::eLess, vk::CullModeFlagBits::eNone);
	}

	FrameGraphResource GradientPass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		VmaAllocator          allocator
	) {
		if (allocator != VK_NULL_HANDLE) {
			lastAllocator = allocator;
		}

		if (allocator != VK_NULL_HANDLE && (currentExtent != extent || bgImage == nullptr)) {
			DestroyTextureResource(allocator);
			currentExtent = extent;

			VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent = VkExtent3D{extent.width, extent.height, 1};
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.format = static_cast<VkFormat>(vk::Format::eR16G16B16A16Sfloat);
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

			VkImage vkImg = VK_NULL_HANDLE;
			if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &bgAllocation, nullptr) == VK_SUCCESS) {
				bgImage = vkImg;

				vk::ImageViewCreateInfo viewInfo{};
				viewInfo.setImage(bgImage);
				viewInfo.setViewType(vk::ImageViewType::e2D);
				viewInfo.setFormat(vk::Format::eR16G16B16A16Sfloat);
				viewInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
				bgImageView = device.createImageView(viewInfo);
			}
		}

		FrameGraphTexture2D texWrapper{bgImage, bgImageView};
		FrameGraphResource importedBg = fg.import("GradientBackground", {extent, vk::Format::eR16G16B16A16Sfloat}, std::move(texWrapper));

		const auto& passData = fg.addCallbackPass<GradientPassData>(
			"GradientPass",
			[&](FrameGraph::Builder& builder, GradientPassData& data) {
				data.target = builder.write(
					importedBg,
					static_cast<uint32_t>(TextureUsage::ColorAttachment)
				);
				builder.setSideEffect();
			},
			[this, extent](const GradientPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             targetTexture = resources.get<FrameGraphTexture2D>(data.target);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTexture.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(
					vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}}
				);

				BeginRendering(cmd, extent, std::span(&colorAttachment, 1));
				cmd.draw(3, 1, 0, 0);
				EndRendering(cmd);
			}
		);
		blackboard.add<GradientPassData>() = passData;
		return passData.target;
	}

} // namespace brassica
