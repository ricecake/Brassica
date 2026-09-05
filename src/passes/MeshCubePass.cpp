#include "passes/MeshCubePass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	MeshCubePass::MeshCubePass(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) : RenderPass(
			"MeshCubePass",
			dev,
			std::array<vk::Format, 3>{vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm},
			vk::Format::eD32Sfloat
		) {
		InitPipeline(instance, dev, globalSet0Layout, watcher);
	}

	MeshCubePass::~MeshCubePass() {
		if (lastAllocator != VK_NULL_HANDLE) {
			DestroyGBufferTextures(lastAllocator);
		}
	}

	void MeshCubePass::DestroyGBufferTextures(VmaAllocator allocator) {
		auto destroyTex = [this, allocator](TextureResource& tex) {
			if (tex.imageView) {
				device.destroyImageView(tex.imageView);
				tex.imageView = nullptr;
			}
			if (tex.image && tex.allocation) {
				vmaDestroyImage(allocator, tex.image, tex.allocation);
				tex.image = nullptr;
				tex.allocation = VK_NULL_HANDLE;
			}
		};

		destroyTex(posTex);
		destroyTex(normTex);
		destroyTex(albTex);
		destroyTex(depthTex);
	}

	void MeshCubePass::CreateGBufferTextures(vk::Extent2D extent, VmaAllocator allocator) {
		DestroyGBufferTextures(allocator);
		currentExtent = extent;

		auto createTex = [this, allocator, extent](vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect, TextureResource& tex) {
			VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent = VkExtent3D{extent.width, extent.height, 1};
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.format = static_cast<VkFormat>(format);
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageInfo.usage = static_cast<VkImageUsageFlags>(usage);
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

			VkImage vkImg = VK_NULL_HANDLE;
			if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &vkImg, &tex.allocation, nullptr) == VK_SUCCESS) {
				tex.image = vkImg;

				vk::ImageViewCreateInfo viewInfo{};
				viewInfo.setImage(tex.image);
				viewInfo.setViewType(vk::ImageViewType::e2D);
				viewInfo.setFormat(format);
				viewInfo.setSubresourceRange(vk::ImageSubresourceRange(aspect, 0, 1, 0, 1));
				tex.imageView = device.createImageView(viewInfo);
			}
		};

		createTex(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, posTex);
		createTex(vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, normTex);
		createTex(vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, albTex);
		createTex(vk::Format::eD32Sfloat, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eDepth, depthTex);
	}

	void MeshCubePass::InitPipeline(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) {
		dls.init(instance, dev);

		if (!meshShader.CompileMeshFromFile(dev, "shaders/cube.mesh")) {
			spdlog::error("Failed to compile cube.mesh shader file");
		}

		if (!fragShader.CompileFragmentFromFile(dev, "shaders/cube.frag")) {
			spdlog::error("Failed to compile cube.frag shader file");
		}

		SetShaders(&meshShader, &fragShader);

		std::array<vk::Format, 3> colorFmts = {
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eR8G8B8A8Unorm
		};

		InitRenderPipeline(
			colorFmts,
			vk::Format::eD32Sfloat,
			std::span(&globalSet0Layout, 1),
			{},
			watcher,
			true,  // depthTestEnable
			true,  // depthWriteEnable
			vk::CompareOp::eLess
		);
	}

	void MeshCubePass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet,
		VmaAllocator          allocator
	) {
		if (allocator != VK_NULL_HANDLE) {
			lastAllocator = allocator;
		}

		if (allocator != VK_NULL_HANDLE && (currentExtent != extent || posTex.image == nullptr)) {
			CreateGBufferTextures(extent, allocator);
		}

		FrameGraphResource importedPos = fg.import("GBuffer_Position", {extent, vk::Format::eR16G16B16A16Sfloat}, FrameGraphTexture2D{posTex.image, posTex.imageView});
		FrameGraphResource importedNorm = fg.import("GBuffer_Normal", {extent, vk::Format::eR16G16B16A16Sfloat}, FrameGraphTexture2D{normTex.image, normTex.imageView});
		FrameGraphResource importedAlb = fg.import("GBuffer_Albedo", {extent, vk::Format::eR8G8B8A8Unorm}, FrameGraphTexture2D{albTex.image, albTex.imageView});
		FrameGraphResource importedDepth = fg.import("GBuffer_Depth", {extent, vk::Format::eD32Sfloat}, FrameGraphTexture2D{depthTex.image, depthTex.imageView});

		const auto& passData = fg.addCallbackPass<MeshCubePassData>(
			"MeshCubePass",
			[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
				data.positionTarget = builder.write(importedPos, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.normalTarget = builder.write(importedNorm, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.albedoTarget = builder.write(importedAlb, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.depthTarget = builder.write(importedDepth, static_cast<uint32_t>(TextureUsage::DepthStencilAttachment));

				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet](const MeshCubePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTexture = resources.get<FrameGraphTexture2D>(data.positionTarget);
				auto& normTexture = resources.get<FrameGraphTexture2D>(data.normalTarget);
				auto& albTexture = resources.get<FrameGraphTexture2D>(data.albedoTarget);
				auto& depthTexture = resources.get<FrameGraphTexture2D>(data.depthTarget);

				std::array<vk::RenderingAttachmentInfo, 3> colorAttachments{};

				// Clear position, normal, albedo to zeros
				for (int i = 0; i < 3; ++i) {
					colorAttachments[i].setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
					colorAttachments[i].setLoadOp(vk::AttachmentLoadOp::eClear);
					colorAttachments[i].setStoreOp(vk::AttachmentStoreOp::eStore);
					colorAttachments[i].setClearValue(vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}});
				}

				colorAttachments[0].setImageView(posTexture.imageView);
				colorAttachments[1].setImageView(normTexture.imageView);
				colorAttachments[2].setImageView(albTexture.imageView);

				vk::RenderingAttachmentInfo depthAttachmentInfo{};
				depthAttachmentInfo.setImageView(depthTexture.imageView);
				depthAttachmentInfo.setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
				depthAttachmentInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
				depthAttachmentInfo.setStoreOp(vk::AttachmentStoreOp::eStore);
				depthAttachmentInfo.setClearValue(vk::ClearValue{vk::ClearDepthStencilValue{1.0f, 0}});

				BeginRendering(cmd, extent, colorAttachments, &depthAttachmentInfo);

				if (globalDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, globalDescriptorSet, nullptr);
				}

				cmd.drawMeshTasksEXT(1, 1, 1, dls);

				EndRendering(cmd);
			}
		);

		blackboard.add<MeshCubePassData>() = passData;
		blackboard.add<GBufferData>() = GBufferData{
			.positionTarget = passData.positionTarget,
			.normalTarget = passData.normalTarget,
			.albedoTarget = passData.albedoTarget,
			.depthTarget = passData.depthTarget
		};
	}

} // namespace brassica
