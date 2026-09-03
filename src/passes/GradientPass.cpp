#include "passes/GradientPass.hpp"

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

	GradientPass::GradientPass(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher) {
		InitPipeline(device, colorFormat, watcher);
	}

	GradientPass::~GradientPass() {
		// Cleanup should be managed via DestroyPipeline
	}

	void GradientPass::InitPipeline(vk::Device device, vk::Format colorFormat, ShaderWatcher* watcher) {
		if (!vertShader.CompileVertexFromFile(device, "shaders/gradient.vert")) {
			spdlog::error("Failed to compile gradient.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(device, "shaders/gradient.frag")) {
			spdlog::error("Failed to compile gradient.frag shader file");
		}

		if (watcher) {
			auto rebuildCb = [this, device, colorFormat]() {
				spdlog::info("Rebuilding GradientPass pipeline due to shader change...");
				device.waitIdle();
				if (pipeline) {
					device.destroyPipeline(pipeline);
					pipeline = nullptr;
				}
				if (pipelineLayout) {
					device.destroyPipelineLayout(pipelineLayout);
					pipelineLayout = nullptr;
				}

				vk::PipelineShaderStageCreateInfo stages[2] = {
					vertShader.GetStageCreateInfo(),
					fragShader.GetStageCreateInfo()
				};

				vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

				vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
				inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
				inputAssembly.setPrimitiveRestartEnable(VK_FALSE);

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

				vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
				try {
					pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
				} catch (const vk::SystemError& err) {
					spdlog::error("Failed to create pipeline layout on reload: {}", err.what());
					return;
				}

				vk::PipelineRenderingCreateInfo renderingCreateInfo{};
				renderingCreateInfo.setColorAttachmentFormats(colorFormat);

				vk::GraphicsPipelineCreateInfo pipelineInfo{};
				pipelineInfo.setPNext(&renderingCreateInfo);
				pipelineInfo.setStages(stages);
				pipelineInfo.setPVertexInputState(&vertexInputInfo);
				pipelineInfo.setPInputAssemblyState(&inputAssembly);
				pipelineInfo.setPViewportState(&viewportState);
				pipelineInfo.setPRasterizationState(&rasterizer);
				pipelineInfo.setPMultisampleState(&multisampling);
				pipelineInfo.setPColorBlendState(&colorBlending);
				pipelineInfo.setPDynamicState(&dynamicState);
				pipelineInfo.setLayout(pipelineLayout);

				auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
				if (result.result == vk::Result::eSuccess) {
					pipeline = result.value;
					spdlog::info("GradientPass pipeline rebuilt successfully.");
				} else {
					spdlog::error("Failed to rebuild graphics pipeline for GradientPass");
				}
			};

			watcher->RegisterShader(&vertShader, rebuildCb);
			watcher->RegisterShader(&fragShader, rebuildCb);
		}

		vk::PipelineShaderStageCreateInfo shaderStages[2] = {
			vertShader.GetStageCreateInfo(),
			fragShader.GetStageCreateInfo()
		};

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
		inputAssembly.setPrimitiveRestartEnable(VK_FALSE);

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

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		try {
			pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
		} catch (const vk::SystemError& err) {
			spdlog::error("Failed to create pipeline layout: {}", err.what());
			return;
		}

		vk::PipelineRenderingCreateInfo renderingCreateInfo{};
		renderingCreateInfo.setColorAttachmentFormats(colorFormat);

		vk::GraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.setPNext(&renderingCreateInfo);
		pipelineInfo.setStages(shaderStages);
		pipelineInfo.setPVertexInputState(&vertexInputInfo);
		pipelineInfo.setPInputAssemblyState(&inputAssembly);
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
			spdlog::error("Failed to create graphics pipeline");
		}
	}

	void GradientPass::DestroyPipeline(vk::Device device) {
		vertShader.Destroy(device);
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

	FrameGraphResource GradientPass::RegisterPass(FrameGraph& fg, FrameGraphResource swapchainImageResource, vk::Extent2D extent) {
		const auto& passData = fg.addCallbackPass<GradientPassData>(
			"GradientPass",
			[&](FrameGraph::Builder& builder, GradientPassData& data) {
				data.target = builder.write(swapchainImageResource);
				builder.setSideEffect();
			},
			[this, extent](const GradientPassData& data, FrameGraphPassResources& resources, void* ctx) {
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
					vk::ClearValue{vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}}
				);

				vk::RenderingInfo renderingInfo{};
				renderingInfo.setRenderArea(vk::Rect2D({0, 0}, extent));
				renderingInfo.setLayerCount(1);
				renderingInfo.setColorAttachments(colorAttachment);

				cmd.beginRendering(renderingInfo);

				cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

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

				cmd.draw(3, 1, 0, 0);

				cmd.endRendering();

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
			}
		);
		return passData.target;
	}

} // namespace brassica
