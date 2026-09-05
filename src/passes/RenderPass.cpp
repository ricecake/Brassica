#include "passes/RenderPass.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	RenderPass::RenderPass(std::string name, vk::Device dev, vk::Format colorFmt, vk::Format depthFmt)
		: Pass(std::move(name), dev), depthFormat(depthFmt) {
		if (colorFmt != vk::Format::eUndefined) {
			colorFormats.push_back(colorFmt);
		}
	}

	RenderPass::RenderPass(std::string name, vk::Device dev, std::span<const vk::Format> colorFmts, vk::Format depthFmt)
		: Pass(std::move(name), dev), colorFormats(colorFmts.begin(), colorFmts.end()), depthFormat(depthFmt) {}

	RenderPass::~RenderPass() {
		DestroyPipeline();
		if (vertOrMeshShader && device) {
			vertOrMeshShader->Destroy(device);
		}
		if (fragShader && device) {
			fragShader->Destroy(device);
		}
	}

	void RenderPass::SetShaders(GraphicsShader* vertexOrMesh, GraphicsShader* fragment) {
		vertOrMeshShader = vertexOrMesh;
		fragShader = fragment;
	}

	void RenderPass::InitRenderPipeline(
		vk::Format                              colorFmt,
		vk::Format                              depthFmt,
		std::span<const vk::DescriptorSetLayout> setLayouts,
		std::span<const vk::PushConstantRange>   pushConstants,
		ShaderWatcher*                          watcher,
		bool                                    enableDepthTest,
		bool                                    enableDepthWrite,
		vk::CompareOp                           depthCompareOp
	) {
		std::vector<vk::Format> fmts;
		if (colorFmt != vk::Format::eUndefined) {
			fmts.push_back(colorFmt);
		}
		InitRenderPipeline(fmts, depthFmt, setLayouts, pushConstants, watcher, enableDepthTest, enableDepthWrite, depthCompareOp);
	}

	void RenderPass::InitRenderPipeline(
		std::span<const vk::Format>              colorFmts,
		vk::Format                              depthFmt,
		std::span<const vk::DescriptorSetLayout> setLayouts,
		std::span<const vk::PushConstantRange>   pushConstants,
		ShaderWatcher*                          watcher,
		bool                                    enableDepthTest,
		bool                                    enableDepthWrite,
		vk::CompareOp                           depthCompareOp
	) {
		colorFormats.assign(colorFmts.begin(), colorFmts.end());
		depthFormat = depthFmt;
		depthTestEnable = enableDepthTest;
		depthWriteEnable = enableDepthWrite;
		this->depthCompareOp = depthCompareOp;

		storedSetLayouts.assign(setLayouts.begin(), setLayouts.end());
		storedPushConstants.assign(pushConstants.begin(), pushConstants.end());

		auto buildPipeline = [this]() {
			if (!vertOrMeshShader || !fragShader) {
				spdlog::error("Cannot build RenderPass pipeline for {}: shaders not set.", name);
				return;
			}

			if (pipeline) {
				device.destroyPipeline(pipeline);
				pipeline = nullptr;
			}
			if (pipelineLayout) {
				device.destroyPipelineLayout(pipelineLayout);
				pipelineLayout = nullptr;
			}

			vk::PipelineLayoutCreateInfo layoutInfo{};
			layoutInfo.setSetLayouts(storedSetLayouts);
			layoutInfo.setPushConstantRanges(storedPushConstants);

			try {
				pipelineLayout = device.createPipelineLayout(layoutInfo);
			} catch (const vk::SystemError& err) {
				spdlog::error("Failed to create pipeline layout for pass {}: {}", name, err.what());
				return;
			}

			vk::PipelineShaderStageCreateInfo stages[2] = {
				vertOrMeshShader->GetStageCreateInfo(),
				fragShader->GetStageCreateInfo()
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
			rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
			rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);

			vk::PipelineMultisampleStateCreateInfo multisampling{};
			multisampling.setSampleShadingEnable(VK_FALSE);
			multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

			std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(colorFormats.size());
			for (size_t i = 0; i < colorFormats.size(); ++i) {
				colorBlendAttachments[i].setColorWriteMask(
					vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
					vk::ColorComponentFlagBits::eA
				);
				colorBlendAttachments[i].setBlendEnable(VK_FALSE);
			}

			vk::PipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.setLogicOpEnable(VK_FALSE);
			colorBlending.setAttachments(colorBlendAttachments);

			vk::PipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.setDepthTestEnable(depthTestEnable ? VK_TRUE : VK_FALSE);
			depthStencil.setDepthWriteEnable(depthWriteEnable ? VK_TRUE : VK_FALSE);
			depthStencil.setDepthCompareOp(this->depthCompareOp);
			depthStencil.setDepthBoundsTestEnable(VK_FALSE);
			depthStencil.setStencilTestEnable(VK_FALSE);

			std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
			vk::PipelineDynamicStateCreateInfo dynamicState{};
			dynamicState.setDynamicStates(dynamicStates);

			vk::PipelineRenderingCreateInfo renderingCreateInfo{};
			if (!colorFormats.empty()) {
				renderingCreateInfo.setColorAttachmentFormats(colorFormats);
			}
			if (depthFormat != vk::Format::eUndefined) {
				renderingCreateInfo.setDepthAttachmentFormat(depthFormat);
			}

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
				spdlog::info("RenderPass '{}' pipeline created/rebuilt successfully.", name);
			} else {
				spdlog::error("Failed to create graphics pipeline for pass '{}'", name);
			}
		};

		if (watcher && vertOrMeshShader && fragShader) {
			auto rebuildCb = [this, buildPipeline]() {
				spdlog::info("Rebuilding RenderPass '{}' pipeline due to shader change...", name);
				device.waitIdle();
				buildPipeline();
			};
			watcher->RegisterShader(vertOrMeshShader, rebuildCb);
			watcher->RegisterShader(fragShader, rebuildCb);
		}

		buildPipeline();
	}

	void RenderPass::DrawMeshTasksIndirectEXT(
		vk::CommandBuffer                cmd,
		vk::Buffer                       buffer,
		vk::DeviceSize                   offset,
		uint32_t                         drawCount,
		uint32_t                         stride,
		const vk::DispatchLoaderDynamic& dls
	) const {
		cmd.drawMeshTasksIndirectEXT(buffer, offset, drawCount, stride, dls);
	}

	void RenderPass::DrawIndexedIndirect(
		vk::CommandBuffer cmd,
		vk::Buffer        buffer,
		vk::DeviceSize    offset,
		uint32_t          drawCount,
		uint32_t          stride
	) const {
		cmd.drawIndexedIndirect(buffer, offset, drawCount, stride);
	}

	void RenderPass::DrawIndirect(
		vk::CommandBuffer cmd,
		vk::Buffer        buffer,
		vk::DeviceSize    offset,
		uint32_t          drawCount,
		uint32_t          stride
	) const {
		cmd.drawIndirect(buffer, offset, drawCount, stride);
	}

	void RenderPass::BeginRendering(
		vk::CommandBuffer                            cmd,
		vk::Extent2D                                 extent,
		std::span<const vk::RenderingAttachmentInfo> colorAttachments,
		const vk::RenderingAttachmentInfo*           depthAttachment
	) const {
		vk::RenderingInfo renderingInfo{};
		renderingInfo.setRenderArea(vk::Rect2D({0, 0}, extent));
		renderingInfo.setLayerCount(1);
		renderingInfo.setColorAttachments(colorAttachments);
		if (depthAttachment) {
			renderingInfo.setPDepthAttachment(depthAttachment);
		}

		cmd.beginRendering(renderingInfo);

		if (pipeline) {
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		}

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
	}

	void RenderPass::EndRendering(vk::CommandBuffer cmd) const {
		cmd.endRendering();
	}

} // namespace brassica
