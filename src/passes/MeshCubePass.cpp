#include "passes/MeshCubePass.hpp"

#include <cstring>
#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	namespace {
		uint32_t FindMemoryType(
			const vk::PhysicalDeviceMemoryProperties& memProperties,
			uint32_t                                  typeFilter,
			vk::MemoryPropertyFlags                   properties
		) {
			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
				if ((typeFilter & (1 << i)) &&
					(memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}
			spdlog::error("Failed to find suitable memory type!");
			return 0;
		}

		void TransitionImageLayout(
			vk::CommandBuffer       cmd,
			vk::Image               image,
			vk::ImageLayout         oldLayout,
			vk::ImageLayout         newLayout,
			vk::PipelineStageFlags2 srcStage,
			vk::PipelineStageFlags2 dstStage,
			vk::AccessFlags2        srcAccess,
			vk::AccessFlags2        dstAccess
		) {
			vk::ImageMemoryBarrier2 barrier{};
			barrier.setSrcStageMask(srcStage);
			barrier.setSrcAccessMask(srcAccess);
			barrier.setDstStageMask(dstStage);
			barrier.setDstAccessMask(dstAccess);
			barrier.setOldLayout(oldLayout);
			barrier.setNewLayout(newLayout);
			barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
			barrier.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			barrier.setImage(image);

			vk::DependencyInfo depInfo{};
			depInfo.setImageMemoryBarriers(barrier);

			cmd.pipelineBarrier2(depInfo);
		}
	} // namespace

	MeshCubePass::MeshCubePass(
		vk::Instance       instance,
		vk::PhysicalDevice physicalDevice,
		vk::Device         device,
		vk::Format         colorFormat,
		ShaderWatcher*     watcher
	) {
		InitPipeline(instance, physicalDevice, device, colorFormat, watcher);
	}

	MeshCubePass::~MeshCubePass() {
		// Cleanup should be managed via DestroyPipeline
	}

	void MeshCubePass::InitPipeline(
		vk::Instance       instance,
		vk::PhysicalDevice physicalDevice,
		vk::Device         device,
		vk::Format         colorFormat,
		ShaderWatcher*     watcher
	) {
		dls.init(instance, device);
		if (!meshShader.CompileMeshFromFile(device, "shaders/cube.mesh")) {
			spdlog::error("Failed to compile cube.mesh shader file");
		}

		if (!fragShader.CompileFragmentFromFile(device, "shaders/cube.frag")) {
			spdlog::error("Failed to compile cube.frag shader file");
		}

		// 1. Descriptor Set Layout
		vk::DescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.setBinding(0);
		layoutBinding.setDescriptorType(vk::DescriptorType::eUniformBuffer);
		layoutBinding.setDescriptorCount(1);
		layoutBinding.setStageFlags(vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment);

		vk::DescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.setBindings(layoutBinding);
		descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);

		// 2. Descriptor Pool & Sets
		vk::DescriptorPoolSize poolSize{};
		poolSize.setType(vk::DescriptorType::eUniformBuffer);
		poolSize.setDescriptorCount(FRAME_OVERLAP);

		vk::DescriptorPoolCreateInfo poolInfo{};
		poolInfo.setMaxSets(FRAME_OVERLAP);
		poolInfo.setPoolSizes(poolSize);
		descriptorPool = device.createDescriptorPool(poolInfo);

		// 3. UBO Buffers
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		std::vector<vk::DescriptorSetLayout> layouts(FRAME_OVERLAP, descriptorSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo{};
		allocInfo.setDescriptorPool(descriptorPool);
		allocInfo.setSetLayouts(layouts);

		auto allocatedSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < FRAME_OVERLAP; i++) {
			descriptorSets[i] = allocatedSets[i];

			vk::BufferCreateInfo bufferInfo{};
			bufferInfo.setSize(sizeof(FrameUBO));
			bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
			bufferInfo.setSharingMode(vk::SharingMode::eExclusive);

			uboBuffers[i] = device.createBuffer(bufferInfo);

			vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(uboBuffers[i]);
			uint32_t memoryTypeIndex = FindMemoryType(
				memProperties,
				memReqs.memoryTypeBits,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
			);

			vk::MemoryAllocateInfo memAllocInfo{memReqs.size, memoryTypeIndex};
			uboMemory[i] = device.allocateMemory(memAllocInfo);
			device.bindBufferMemory(uboBuffers[i], uboMemory[i], 0);

			uboMapped[i] = device.mapMemory(uboMemory[i], 0, sizeof(FrameUBO));

			vk::DescriptorBufferInfo bufferDescInfo{};
			bufferDescInfo.setBuffer(uboBuffers[i]);
			bufferDescInfo.setOffset(0);
			bufferDescInfo.setRange(sizeof(FrameUBO));

			vk::WriteDescriptorSet descriptorWrite{};
			descriptorWrite.setDstSet(descriptorSets[i]);
			descriptorWrite.setDstBinding(0);
			descriptorWrite.setDstArrayElement(0);
			descriptorWrite.setDescriptorType(vk::DescriptorType::eUniformBuffer);
			descriptorWrite.setBufferInfo(bufferDescInfo);

			device.updateDescriptorSets(descriptorWrite, nullptr);
		}

		// 4. Pipeline Layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.setSetLayouts(descriptorSetLayout);
		pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

		// 5. Shader Watcher setup
		if (watcher) {
			auto rebuildCb = [this, device, colorFormat]() {
				spdlog::info("Rebuilding MeshCubePass pipeline due to shader change...");
				device.waitIdle();
				if (pipeline) {
					device.destroyPipeline(pipeline);
					pipeline = nullptr;
				}

				vk::PipelineShaderStageCreateInfo stages[2] = {
					meshShader.GetStageCreateInfo(),
					fragShader.GetStageCreateInfo()
				};

				vk::PipelineViewportStateCreateInfo viewportState{};
				viewportState.setViewportCount(1);
				viewportState.setScissorCount(1);

				vk::PipelineRasterizationStateCreateInfo rasterizer{};
				rasterizer.setDepthClampEnable(VK_FALSE);
				rasterizer.setRasterizerDiscardEnable(VK_FALSE);
				rasterizer.setPolygonMode(vk::PolygonMode::eFill);
				rasterizer.setLineWidth(1.0f);
				rasterizer.setCullMode(vk::CullModeFlagBits::eNone);
				rasterizer.setFrontFace(vk::FrontFace::eClockwise);

				vk::PipelineMultisampleStateCreateInfo multisampling{};
				multisampling.setSampleShadingEnable(VK_FALSE);
				multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

				vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
				colorBlendAttachment.setColorWriteMask(
					vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
					vk::ColorComponentFlagBits::eA
				);
				colorBlendAttachment.setBlendEnable(VK_FALSE);

				vk::PipelineColorBlendStateCreateInfo colorBlending{};
				colorBlending.setLogicOpEnable(VK_FALSE);
				colorBlending.setAttachments(colorBlendAttachment);

				std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
				vk::PipelineDynamicStateCreateInfo dynamicState{};
				dynamicState.setDynamicStates(dynamicStates);

				vk::PipelineRenderingCreateInfo renderingCreateInfo{};
				renderingCreateInfo.setColorAttachmentFormats(colorFormat);

				vk::GraphicsPipelineCreateInfo pipelineInfo{};
				pipelineInfo.setPNext(&renderingCreateInfo);
				pipelineInfo.setStages(stages);
				pipelineInfo.setPVertexInputState(nullptr);
				pipelineInfo.setPInputAssemblyState(nullptr);
				pipelineInfo.setPViewportState(&viewportState);
				pipelineInfo.setPRasterizationState(&rasterizer);
				pipelineInfo.setPMultisampleState(&multisampling);
				pipelineInfo.setPColorBlendState(&colorBlending);
				pipelineInfo.setPDynamicState(&dynamicState);
				pipelineInfo.setLayout(pipelineLayout);

				auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
				if (result.result == vk::Result::eSuccess) {
					pipeline = result.value;
					spdlog::info("MeshCubePass pipeline rebuilt successfully.");
				} else {
					spdlog::error("Failed to rebuild graphics pipeline for MeshCubePass");
				}
			};

			watcher->RegisterShader(&meshShader, rebuildCb);
			watcher->RegisterShader(&fragShader, rebuildCb);
		}

		// 6. Graphics Pipeline Creation
		vk::PipelineShaderStageCreateInfo shaderStages[2] = {
			meshShader.GetStageCreateInfo(),
			fragShader.GetStageCreateInfo()
		};

		vk::PipelineViewportStateCreateInfo viewportState{};
		viewportState.setViewportCount(1);
		viewportState.setScissorCount(1);

		vk::PipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.setDepthClampEnable(VK_FALSE);
		rasterizer.setRasterizerDiscardEnable(VK_FALSE);
		rasterizer.setPolygonMode(vk::PolygonMode::eFill);
		rasterizer.setLineWidth(1.0f);
		rasterizer.setCullMode(vk::CullModeFlagBits::eNone);
		rasterizer.setFrontFace(vk::FrontFace::eClockwise);

		vk::PipelineMultisampleStateCreateInfo multisampling{};
		multisampling.setSampleShadingEnable(VK_FALSE);
		multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.setColorWriteMask(
			vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA
		);
		colorBlendAttachment.setBlendEnable(VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.setLogicOpEnable(VK_FALSE);
		colorBlending.setAttachments(colorBlendAttachment);

		std::vector<vk::DynamicState>      dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.setDynamicStates(dynamicStates);

		vk::PipelineRenderingCreateInfo renderingCreateInfo{};
		renderingCreateInfo.setColorAttachmentFormats(colorFormat);

		vk::GraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.setPNext(&renderingCreateInfo);
		pipelineInfo.setStages(shaderStages);
		pipelineInfo.setPVertexInputState(nullptr);
		pipelineInfo.setPInputAssemblyState(nullptr);
		pipelineInfo.setPViewportState(&viewportState);
		pipelineInfo.setPRasterizationState(&rasterizer);
		pipelineInfo.setPMultisampleState(&multisampling);
		pipelineInfo.setPColorBlendState(&colorBlending);
		pipelineInfo.setPDynamicState(&dynamicState);
		pipelineInfo.setLayout(pipelineLayout);

		auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
		if (result.result == vk::Result::eSuccess) {
			pipeline = result.value;
		} else {
			spdlog::error("Failed to create graphics pipeline for MeshCubePass");
		}
	}

	void MeshCubePass::DestroyPipeline(vk::Device device) {
		meshShader.Destroy(device);
		fragShader.Destroy(device);

		for (size_t i = 0; i < FRAME_OVERLAP; i++) {
			if (uboMapped[i] && uboMemory[i]) {
				device.unmapMemory(uboMemory[i]);
				uboMapped[i] = nullptr;
			}
			if (uboBuffers[i]) {
				device.destroyBuffer(uboBuffers[i]);
				uboBuffers[i] = nullptr;
			}
			if (uboMemory[i]) {
				device.freeMemory(uboMemory[i]);
				uboMemory[i] = nullptr;
			}
		}

		if (descriptorPool) {
			device.destroyDescriptorPool(descriptorPool);
			descriptorPool = nullptr;
		}
		if (descriptorSetLayout) {
			device.destroyDescriptorSetLayout(descriptorSetLayout);
			descriptorSetLayout = nullptr;
		}
		if (pipeline) {
			device.destroyPipeline(pipeline);
			pipeline = nullptr;
		}
		if (pipelineLayout) {
			device.destroyPipelineLayout(pipelineLayout);
			pipelineLayout = nullptr;
		}
	}

	void MeshCubePass::RegisterPass(
		FrameGraph&        fg,
		FrameGraphResource swapchainImageResource,
		vk::Extent2D       extent,
		const FrameUBO&    uboData,
		uint32_t           frameIndex
	) {
		uint32_t activeFrame = frameIndex % FRAME_OVERLAP;
		if (uboMapped[activeFrame]) {
			std::memcpy(uboMapped[activeFrame], &uboData, sizeof(FrameUBO));
		}

		fg.addCallbackPass<MeshCubePassData>(
			"MeshCubePass",
			[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
				data.target = builder.write(swapchainImageResource);
				builder.setSideEffect();
			},
			[this, extent, activeFrame](const MeshCubePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             targetTexture = resources.get<FrameGraphTexture>(data.target);

				TransitionImageLayout(
					cmd,
					targetTexture.image,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					{},
					vk::AccessFlagBits2::eColorAttachmentWrite
				);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTexture.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eClear);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);
				colorAttachment.setClearValue(
					vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.05f, 0.05f, 0.08f, 1.0f}}}
				);

				vk::RenderingInfo renderingInfo{};
				renderingInfo.setRenderArea(vk::Rect2D({0, 0}, extent));
				renderingInfo.setLayerCount(1);
				renderingInfo.setColorAttachments(colorAttachment);

				cmd.beginRendering(renderingInfo);

				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

				cmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pipelineLayout,
					0,
					descriptorSets[activeFrame],
					nullptr
				);

				vk::Viewport viewport{
					0.0f,
					0.0f,
					static_cast<float>(extent.width),
					static_cast<float>(extent.height),
					0.0f,
					1.0f
				};
				cmd.setViewport(0, viewport);

				vk::Rect2D scissor{{0, 0}, extent};
				cmd.setScissor(0, scissor);

				cmd.drawMeshTasksEXT(1, 1, 1, dls);

				cmd.endRendering();

				TransitionImageLayout(
					cmd,
					targetTexture.image,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::ImageLayout::ePresentSrcKHR,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::PipelineStageFlagBits2::eBottomOfPipe,
					vk::AccessFlagBits2::eColorAttachmentWrite,
					{}
				);
			}
		);
	}

} // namespace brassica
