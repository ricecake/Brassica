#include "passes/TerrainPass.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	TerrainPass::TerrainPass(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) : RenderPass(
			"TerrainPass",
			dev,
			std::array<vk::Format, 3>{vk::Format::eR16G16B16A16Sfloat, vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm},
			vk::Format::eD32Sfloat
		) {
		InitPipeline(instance, dev, globalSet0Layout, watcher);
	}

	TerrainPass::~TerrainPass() {
		if (terrainDescriptorPool) {
			device.destroyDescriptorPool(terrainDescriptorPool);
			terrainDescriptorPool = nullptr;
		}
		if (terrainSet1Layout) {
			device.destroyDescriptorSetLayout(terrainSet1Layout);
			terrainSet1Layout = nullptr;
		}
		if (taskShader.GetModule()) {
			taskShader.Destroy(device);
		}
		if (lastAllocator != VK_NULL_HANDLE) {
			DestroyGBufferTextures(lastAllocator);
		}
	}

	void TerrainPass::DestroyGBufferTextures(VmaAllocator allocator) {
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

	void TerrainPass::CreateGBufferTextures(vk::Extent2D extent, VmaAllocator allocator) {
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

	void TerrainPass::InitPipeline(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) {
		dls.init(instance, dev);

		// Set 1 Layout for clipmap texture sampler
		vk::DescriptorSetLayoutBinding samplerBinding{};
		samplerBinding.setBinding(0);
		samplerBinding.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		samplerBinding.setDescriptorCount(1);
		samplerBinding.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(samplerBinding);
		terrainSet1Layout = dev.createDescriptorSetLayout(layoutInfo);

		// Descriptor Pool for Set 1
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eCombinedImageSampler);
		poolSize.setDescriptorCount(1);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(1);
		poolInfo.setPoolSizes(poolSize);
		terrainDescriptorPool = dev.createDescriptorPool(poolInfo);

		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(terrainDescriptorPool);
		allocInfo.setSetLayouts(terrainSet1Layout);
		terrainDescriptorSet = dev.allocateDescriptorSets(allocInfo).front();

		InitPipelineCustom(instance, dev, globalSet0Layout, watcher);
	}

	void TerrainPass::InitPipelineCustom(
		vk::Instance            instance,
		vk::Device              dev,
		vk::DescriptorSetLayout globalSet0Layout,
		ShaderWatcher*          watcher
	) {
		if (!taskShader.CompileTaskFromFile(dev, "shaders/terrain.task")) {
			spdlog::error("Failed to compile terrain.task shader file");
		}
		if (!meshShader.CompileMeshFromFile(dev, "shaders/terrain.mesh")) {
			spdlog::error("Failed to compile terrain.mesh shader file");
		}
		if (!fragShader.CompileFragmentFromFile(dev, "shaders/terrain.frag")) {
			spdlog::error("Failed to compile terrain.frag shader file");
		}

		vertOrMeshShader = &meshShader;
		RenderPass::fragShader = &this->fragShader;

		std::array<vk::DescriptorSetLayout, 2> setLayouts = {globalSet0Layout, terrainSet1Layout};
		vk::PushConstantRange pushConstantRange{};
		pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT);
		pushConstantRange.setOffset(0);
		pushConstantRange.setSize(sizeof(TerrainPushConstants));

		storedSetLayouts.assign(setLayouts.begin(), setLayouts.end());
		storedPushConstants.assign({pushConstantRange});

		auto buildPipeline = [this]() {
			if (pipeline) device.destroyPipeline(pipeline);
			if (pipelineLayout) device.destroyPipelineLayout(pipelineLayout);

			vk::PipelineLayoutCreateInfo layoutInfo{};
			layoutInfo.setSetLayouts(storedSetLayouts);
			layoutInfo.setPushConstantRanges(storedPushConstants);
			pipelineLayout = device.createPipelineLayout(layoutInfo);

			std::vector<vk::PipelineShaderStageCreateInfo> stages = {
				taskShader.GetStageCreateInfo(),
				meshShader.GetStageCreateInfo(),
				fragShader.GetStageCreateInfo()
			};

			vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
			vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);

			vk::PipelineViewportStateCreateInfo viewportState{};
			viewportState.setViewportCount(1);
			viewportState.setScissorCount(1);

			vk::PipelineRasterizationStateCreateInfo rasterizer{};
			rasterizer.setPolygonMode(vk::PolygonMode::eFill);
			rasterizer.setLineWidth(1.0f);
			rasterizer.setCullMode(vk::CullModeFlagBits::eNone); // Render terrain double-sided / front
			rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);

			vk::PipelineMultisampleStateCreateInfo multisampling{};
			multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

			std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(colorFormats.size());
			for (size_t i = 0; i < colorFormats.size(); ++i) {
				colorBlendAttachments[i].setColorWriteMask(
					vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
				);
			}

			vk::PipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.setAttachments(colorBlendAttachments);

			vk::PipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.setDepthTestEnable(VK_TRUE);
			depthStencil.setDepthWriteEnable(VK_TRUE);
			depthStencil.setDepthCompareOp(vk::CompareOp::eLess);

			std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
			vk::PipelineDynamicStateCreateInfo dynamicState{};
			dynamicState.setDynamicStates(dynamicStates);

			vk::PipelineRenderingCreateInfo renderingCreateInfo{};
			renderingCreateInfo.setColorAttachmentFormats(colorFormats);
			renderingCreateInfo.setDepthAttachmentFormat(depthFormat);

			vk::GraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.setPNext(&renderingCreateInfo);
			pipelineInfo.setStages(stages);
			pipelineInfo.setPVertexInputState(&vertexInputInfo);
			pipelineInfo.setPInputAssemblyState(&inputAssembly);
			pipelineInfo.setPViewportState(&viewportState);
			pipelineInfo.setPRasterizationState(&rasterizer);
			pipelineInfo.setPMultisampleState(&multisampling);
			pipelineInfo.setPColorBlendState(&colorBlending);
			pipelineInfo.setPDepthStencilState(&depthStencil);
			pipelineInfo.setPDynamicState(&dynamicState);
			pipelineInfo.setLayout(pipelineLayout);

			auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
			if (result.result == vk::Result::eSuccess) {
				pipeline = result.value;
				spdlog::info("TerrainPass pipeline created successfully.");
			} else {
				spdlog::error("Failed to create TerrainPass graphics pipeline.");
			}
		};

		if (watcher) {
			auto rebuildCb = [this, buildPipeline]() {
				spdlog::info("Rebuilding TerrainPass pipeline due to shader modification...");
				device.waitIdle();
				buildPipeline();
			};
			watcher->RegisterShader(&taskShader, rebuildCb);
			watcher->RegisterShader(&meshShader, rebuildCb);
			watcher->RegisterShader(&fragShader, rebuildCb);
		}

		buildPipeline();
	}

	void TerrainPass::UpdateClipmapDescriptor(vk::ImageView clipmapImageView, vk::Sampler clipmapSampler) {
		if (!terrainDescriptorSet || !clipmapImageView || !clipmapSampler) return;

		vk::DescriptorImageInfo imageInfo{};
		imageInfo.setImageView(clipmapImageView);
		imageInfo.setSampler(clipmapSampler);
		imageInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);

		vk::WriteDescriptorSet descriptorWrite{};
		descriptorWrite.setDstSet(terrainDescriptorSet);
		descriptorWrite.setDstBinding(0);
		descriptorWrite.setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
		descriptorWrite.setImageInfo(imageInfo);

		device.updateDescriptorSets(descriptorWrite, nullptr);
	}

	void TerrainPass::RegisterPass(
		FrameGraph&           fg,
		FrameGraphBlackboard& blackboard,
		vk::Extent2D          extent,
		vk::DescriptorSet     globalDescriptorSet,
		const TerrainPushConstants& pushConstants,
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

		const auto& passData = fg.addCallbackPass<TerrainPassData>(
			"TerrainPass",
			[&](FrameGraph::Builder& builder, TerrainPassData& data) {
				data.positionTarget = builder.write(importedPos, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.normalTarget = builder.write(importedNorm, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.albedoTarget = builder.write(importedAlb, static_cast<uint32_t>(TextureUsage::ColorAttachment));
				data.depthTarget = builder.write(importedDepth, static_cast<uint32_t>(TextureUsage::DepthStencilAttachment));

				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet, pushConstants](const TerrainPassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);

				auto& posTexture = resources.get<FrameGraphTexture2D>(data.positionTarget);
				auto& normTexture = resources.get<FrameGraphTexture2D>(data.normalTarget);
				auto& albTexture = resources.get<FrameGraphTexture2D>(data.albedoTarget);
				auto& depthTexture = resources.get<FrameGraphTexture2D>(data.depthTarget);

				std::array<vk::RenderingAttachmentInfo, 3> colorAttachments{};

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
				if (terrainDescriptorSet) {
					cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, terrainDescriptorSet, nullptr);
				}

				cmd.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(TerrainPushConstants), &pushConstants);

				// Dispatch task groups: ceil(totalMeshlets / 32)
				uint32_t taskGroupCount = (pushConstants.gridParams.z + 31) / 32;
				cmd.drawMeshTasksEXT(taskGroupCount, 1, 1, dls);

				EndRendering(cmd);
			}
		);

		blackboard.add<TerrainPassData>() = passData;
		blackboard.add<GBufferData>() = GBufferData{
			.positionTarget = passData.positionTarget,
			.normalTarget = passData.normalTarget,
			.albedoTarget = passData.albedoTarget,
			.depthTarget = passData.depthTarget
		};
	}

} // namespace brassica
