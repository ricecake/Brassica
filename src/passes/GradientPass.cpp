#include "passes/GradientPass.hpp"

#include <vector>
#include "spdlog/spdlog.h"

namespace brassica {

	namespace {
		void TransitionImageLayout(
			VkCommandBuffer      cmd,
			VkImage              image,
			VkImageLayout        oldLayout,
			VkImageLayout        newLayout,
			VkPipelineStageFlags srcStage,
			VkPipelineStageFlags dstStage,
			VkAccessFlags        srcAccess,
			VkAccessFlags        dstAccess
		) {
			VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
			barrier.srcStageMask = srcStage;
			barrier.srcAccessMask = srcAccess;
			barrier.dstStageMask = dstStage;
			barrier.dstAccessMask = dstAccess;
			barrier.oldLayout = oldLayout;
			barrier.newLayout = newLayout;
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.image = image;

			VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
			depInfo.imageMemoryBarrierCount = 1;
			depInfo.pImageMemoryBarriers = &barrier;

			vkCmdPipelineBarrier2(cmd, &depInfo);
		}
	} // namespace

	GradientPass::GradientPass(VkDevice device, VkFormat colorFormat) {
		InitPipeline(device, colorFormat);
	}

	GradientPass::~GradientPass() {
		// Cleanup should be managed via DestroyPipeline
	}

	void GradientPass::InitPipeline(VkDevice device, VkFormat colorFormat) {
		if (!vertShader.CompileVertexFromFile(device, "shaders/gradient.vert")) {
			spdlog::error("Failed to compile gradient.vert shader file");
		}

		if (!fragShader.CompileFragmentFromFile(device, "shaders/gradient.frag")) {
			spdlog::error("Failed to compile gradient.frag shader file");
		}

		VkPipelineShaderStageCreateInfo shaderStages[2]{
			vertShader.GetStageCreateInfo(),
			fragShader.GetStageCreateInfo()
		};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkDynamicState                   dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
		VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
		dynamicState.dynamicStateCount = 2;
		dynamicState.pDynamicStates = dynamicStates;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

		VkPipelineRenderingCreateInfo renderingCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		renderingCreateInfo.colorAttachmentCount = 1;
		renderingCreateInfo.pColorAttachmentFormats = &colorFormat;

		VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		pipelineInfo.pNext = &renderingCreateInfo;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = VK_NULL_HANDLE;

		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

	}

	void GradientPass::DestroyPipeline(VkDevice device) {
		vertShader.Destroy(device);
		fragShader.Destroy(device);

		if (pipeline) {
			vkDestroyPipeline(device, pipeline, nullptr);
			pipeline = VK_NULL_HANDLE;
		}
		if (pipelineLayout) {
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			pipelineLayout = VK_NULL_HANDLE;
		}
	}

	void GradientPass::RegisterPass(FrameGraph& fg, FrameGraphResource swapchainImageResource, VkExtent2D extent) {
		fg.addCallbackPass<GradientPassData>(
			"GradientPass",
			[&](FrameGraph::Builder& builder, GradientPassData& data) {
				data.target = builder.write(swapchainImageResource);
				builder.setSideEffect();
			},
			[this, extent](const GradientPassData& data, FrameGraphPassResources& resources, void* ctx) {
				VkCommandBuffer cmd = *static_cast<VkCommandBuffer*>(ctx);
				auto&           targetTexture = resources.get<FrameGraphTexture>(data.target);

				TransitionImageLayout(
					cmd,
					targetTexture.image,
					VK_IMAGE_LAYOUT_UNDEFINED,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					0,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
				);

				VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
				colorAttachment.imageView = targetTexture.imageView;
				colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
				colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

				VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
				renderingInfo.renderArea = {{0, 0}, extent};
				renderingInfo.layerCount = 1;
				renderingInfo.colorAttachmentCount = 1;
				renderingInfo.pColorAttachments = &colorAttachment;

				vkCmdBeginRendering(cmd, &renderingInfo);

				vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

				VkViewport viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
				vkCmdSetViewport(cmd, 0, 1, &viewport);

				VkRect2D scissor{{0, 0}, extent};
				vkCmdSetScissor(cmd, 0, 1, &scissor);

				vkCmdDraw(cmd, 3, 1, 0, 0);

				vkCmdEndRendering(cmd);

				TransitionImageLayout(
					cmd,
					targetTexture.image,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
					VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
					0
				);
			}
		);
	}

} // namespace brassica
