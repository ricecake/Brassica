#include "passes/MeshCubePass.hpp"

#include <vector>

#include "spdlog/spdlog.h"
#include "ShaderWatcher.hpp"

namespace brassica {

	namespace {
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
		vk::Instance            instance,
		vk::Device              device,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFormat,
		ShaderWatcher*          watcher
	) {
		InitPipeline(instance, device, globalSet0Layout, colorFormat, watcher);
	}

	MeshCubePass::~MeshCubePass() {
		// Cleanup should be managed via DestroyPipeline
	}

	void MeshCubePass::InitPipeline(
		vk::Instance            instance,
		vk::Device              device,
		vk::DescriptorSetLayout globalSet0Layout,
		vk::Format              colorFormat,
		ShaderWatcher*          watcher
	) {
		dls.init(instance, device);

		if (!meshShader.CompileMeshFromFile(device, "shaders/cube.mesh")) {
			spdlog::error("Failed to compile cube.mesh shader file");
		}

		if (!fragShader.CompileFragmentFromFile(device, "shaders/cube.frag")) {
			spdlog::error("Failed to compile cube.frag shader file");
		}

		// 1. Pipeline Layout using global Set 0 Layout
		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.setSetLayouts(globalSet0Layout);
		pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

		// 2. Shader Watcher setup
		if (watcher) {
			auto rebuildCb = [this, device, colorFormat]() {
				spdlog::info("Rebuilding MeshCubePass pipeline due to shader change...");
				device.waitIdle();
				if (pipeline) {
					device.destroyPipeline(pipeline);
					pipeline = nullptr;
				}

				meshShader.CompileMeshFromFile(device, "shaders/cube.mesh");
				fragShader.CompileFragmentFromFile(device, "shaders/cube.frag");

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
				rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
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

		// 3. Graphics Pipeline Creation
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
		rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
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

		if (pipeline) {
			device.destroyPipeline(pipeline);
			pipeline = nullptr;
		}
		if (pipelineLayout) {
			device.destroyPipelineLayout(pipelineLayout);
			pipelineLayout = nullptr;
		}
	}

	FrameGraphResource MeshCubePass::RegisterPass(
		FrameGraph&        fg,
		FrameGraphResource inputResource,
		vk::Extent2D       extent,
		vk::DescriptorSet  globalDescriptorSet
	) {
		const auto& passData = fg.addCallbackPass<MeshCubePassData>(
			"MeshCubePass",
			[&](FrameGraph::Builder& builder, MeshCubePassData& data) {
				data.target = builder.write(inputResource);
				builder.setSideEffect();
			},
			[this, extent, globalDescriptorSet](const MeshCubePassData& data, FrameGraphPassResources& resources, void* ctx) {
				vk::CommandBuffer cmd = *static_cast<vk::CommandBuffer*>(ctx);
				auto&             targetTexture = resources.get<FrameGraphTexture>(data.target);

				TransitionImageLayout(
					cmd,
					targetTexture.image,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::ImageLayout::eColorAttachmentOptimal,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::PipelineStageFlagBits2::eColorAttachmentOutput,
					vk::AccessFlagBits2::eColorAttachmentWrite,
					vk::AccessFlagBits2::eColorAttachmentWrite
				);

				vk::RenderingAttachmentInfo colorAttachment{};
				colorAttachment.setImageView(targetTexture.imageView);
				colorAttachment.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
				colorAttachment.setLoadOp(vk::AttachmentLoadOp::eLoad);
				colorAttachment.setStoreOp(vk::AttachmentStoreOp::eStore);

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
					globalDescriptorSet,
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
		return passData.target;
	}

} // namespace brassica
