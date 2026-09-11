#include "render/PipelineLibrary.hpp"

#include <array>
#include <vector>

#include "spdlog/spdlog.h"

namespace brassica::render {

	ResolvedPipeline PipelineLibrary::Resolve(const GraphicsPipelineRequest& request) const {
		return Build(request);
	}

	ResolvedPipeline PipelineLibrary::Resolve(const ComputePipelineRequest& request) const {
		return Build(request);
	}

	ResolvedPipeline PipelineLibrary::ResolveCached(const GraphicsPipelineRequest& request) {
		CacheKey key = BuildKey(request);
		if (auto it = m_graphicsCache.find(key); it != m_graphicsCache.end()) {
			return it->second;
		}
		ResolvedPipeline resolved = Build(request);
		m_graphicsCache.emplace(std::move(key), resolved);
		return resolved;
	}

	ResolvedPipeline PipelineLibrary::ResolveCached(const ComputePipelineRequest& request) {
		CacheKey key = BuildKey(request);
		if (auto it = m_computeCache.find(key); it != m_computeCache.end()) {
			return it->second;
		}
		ResolvedPipeline resolved = Build(request);
		m_computeCache.emplace(std::move(key), resolved);
		return resolved;
	}

	void PipelineLibrary::Reset() {
		for (auto& [key, resolved] : m_graphicsCache) {
			if (resolved.pipeline) {
				m_device.destroyPipeline(resolved.pipeline);
			}
			if (resolved.layout) {
				m_device.destroyPipelineLayout(resolved.layout);
			}
		}
		m_graphicsCache.clear();

		for (auto& [key, resolved] : m_computeCache) {
			if (resolved.pipeline) {
				m_device.destroyPipeline(resolved.pipeline);
			}
			if (resolved.layout) {
				m_device.destroyPipelineLayout(resolved.layout);
			}
		}
		m_computeCache.clear();
	}

	PipelineLibrary::CacheKey PipelineLibrary::BuildKey(const GraphicsPipelineRequest& request) {
		CacheKey key;
		key.words.push_back(request.stages.size());
		for (GraphicsShader* shader : request.stages) {
			key.words.push_back(reinterpret_cast<std::uint64_t>(shader));
			key.words.push_back(shader ? shader->GetGeneration() : 0);
		}

		key.words.push_back(static_cast<std::uint64_t>(static_cast<VkCullModeFlags>(request.state.cullMode)));
		key.words.push_back(request.state.depthTest ? 1 : 0);
		key.words.push_back(request.state.depthWrite ? 1 : 0);
		key.words.push_back(static_cast<std::uint64_t>(request.state.depthCompareOp));
		key.words.push_back(request.state.enableBlend ? 1 : 0);
		key.words.push_back(request.state.enableShadingRate ? 1 : 0);

		key.words.push_back(request.colorFormats.size());
		for (vk::Format fmt : request.colorFormats) {
			key.words.push_back(static_cast<std::uint64_t>(fmt));
		}
		key.words.push_back(static_cast<std::uint64_t>(request.depthFormat));

		key.words.push_back(request.setLayouts.size());
		for (vk::DescriptorSetLayout layout : request.setLayouts) {
			key.words.push_back(reinterpret_cast<std::uint64_t>(static_cast<VkDescriptorSetLayout>(layout)));
		}

		key.words.push_back(request.pushConstantRanges.size());
		for (const vk::PushConstantRange& range : request.pushConstantRanges) {
			key.words.push_back(static_cast<std::uint64_t>(static_cast<VkShaderStageFlags>(range.stageFlags)));
			key.words.push_back(range.offset);
			key.words.push_back(range.size);
		}

		return key;
	}

	PipelineLibrary::CacheKey PipelineLibrary::BuildKey(const ComputePipelineRequest& request) {
		CacheKey key;
		key.words.push_back(reinterpret_cast<std::uint64_t>(request.shader));
		key.words.push_back(request.shader ? request.shader->GetGeneration() : 0);

		key.words.push_back(request.setLayouts.size());
		for (vk::DescriptorSetLayout layout : request.setLayouts) {
			key.words.push_back(reinterpret_cast<std::uint64_t>(static_cast<VkDescriptorSetLayout>(layout)));
		}

		key.words.push_back(request.pushConstantRanges.size());
		for (const vk::PushConstantRange& range : request.pushConstantRanges) {
			key.words.push_back(static_cast<std::uint64_t>(static_cast<VkShaderStageFlags>(range.stageFlags)));
			key.words.push_back(range.offset);
			key.words.push_back(range.size);
		}

		return key;
	}

	ResolvedPipeline PipelineLibrary::Build(const GraphicsPipelineRequest& request) const {
		ResolvedPipeline resolved{};

		vk::PipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.setSetLayouts(request.setLayouts);
		layoutInfo.setPushConstantRanges(request.pushConstantRanges);
		try {
			resolved.layout = m_device.createPipelineLayout(layoutInfo);
		} catch (const vk::SystemError& err) {
			spdlog::error("PipelineLibrary: failed to create graphics pipeline layout: {}", err.what());
			return resolved;
		}

		std::vector<vk::PipelineShaderStageCreateInfo> stages;
		stages.reserve(request.stages.size());
		for (GraphicsShader* shader : request.stages) {
			if (shader) {
				stages.push_back(shader->GetStageCreateInfo());
			}
		}

		// Identical in every graphics pipeline this engine builds -- the whole point of this
		// function existing is that these live exactly once.
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
		rasterizer.setCullMode(request.state.cullMode);
		rasterizer.setFrontFace(vk::FrontFace::eCounterClockwise);

		vk::PipelineMultisampleStateCreateInfo multisampling{};
		multisampling.setSampleShadingEnable(VK_FALSE);
		multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);

		std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(request.colorFormats.size());
		for (auto& attachment : colorBlendAttachments) {
			attachment.setColorWriteMask(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA
			);
			if (request.state.enableBlend) {
				// Standard alpha blend -- the one config every current consumer of this flag
				// (the water branch) actually wants; a second blend preset can be added if a
				// future pass needs a different one.
				attachment.setBlendEnable(VK_TRUE);
				attachment.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha);
				attachment.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha);
				attachment.setColorBlendOp(vk::BlendOp::eAdd);
				attachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
				attachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
				attachment.setAlphaBlendOp(vk::BlendOp::eAdd);
			} else {
				attachment.setBlendEnable(VK_FALSE);
			}
		}

		vk::PipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.setLogicOpEnable(VK_FALSE);
		colorBlending.setAttachments(colorBlendAttachments);

		vk::PipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.setDepthTestEnable(request.state.depthTest ? VK_TRUE : VK_FALSE);
		depthStencil.setDepthWriteEnable(request.state.depthWrite ? VK_TRUE : VK_FALSE);
		depthStencil.setDepthCompareOp(request.state.depthCompareOp);
		depthStencil.setDepthBoundsTestEnable(VK_FALSE);
		depthStencil.setStencilTestEnable(VK_FALSE);

		std::array<vk::DynamicState, 2>    dynamicStates{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.setDynamicStates(dynamicStates);

		VkPipelineFragmentShadingRateStateCreateInfoKHR shadingRateState{};
		shadingRateState.sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
		shadingRateState.fragmentSize = {1, 1};
		shadingRateState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;
		shadingRateState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR;

		vk::PipelineRenderingCreateInfo renderingCreateInfo{};
		if (!request.colorFormats.empty()) {
			renderingCreateInfo.setColorAttachmentFormats(request.colorFormats);
		}
		if (request.depthFormat != vk::Format::eUndefined) {
			renderingCreateInfo.setDepthAttachmentFormat(request.depthFormat);
		}
		if (request.state.enableShadingRate) {
			renderingCreateInfo.setPNext(&shadingRateState);
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
		pipelineInfo.setLayout(resolved.layout);

		auto result = m_device.createGraphicsPipeline(m_cache, pipelineInfo);
		if (result.result == vk::Result::eSuccess) {
			resolved.pipeline = result.value;
		} else {
			spdlog::error("PipelineLibrary: failed to create graphics pipeline.");
		}
		return resolved;
	}

	ResolvedPipeline PipelineLibrary::Build(const ComputePipelineRequest& request) const {
		ResolvedPipeline resolved{};

		if (!request.shader) {
			spdlog::error("PipelineLibrary: cannot resolve a compute pipeline with no shader set.");
			return resolved;
		}

		vk::PipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.setSetLayouts(request.setLayouts);
		layoutInfo.setPushConstantRanges(request.pushConstantRanges);
		try {
			resolved.layout = m_device.createPipelineLayout(layoutInfo);
		} catch (const vk::SystemError& err) {
			spdlog::error("PipelineLibrary: failed to create compute pipeline layout: {}", err.what());
			return resolved;
		}

		vk::ComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.setStage(request.shader->GetStageCreateInfo());
		pipelineInfo.setLayout(resolved.layout);

		auto result = m_device.createComputePipeline(m_cache, pipelineInfo);
		if (result.result == vk::Result::eSuccess) {
			resolved.pipeline = result.value;
		} else {
			spdlog::error("PipelineLibrary: failed to create compute pipeline.");
		}
		return resolved;
	}

} // namespace brassica::render
