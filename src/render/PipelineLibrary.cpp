#include "render/PipelineLibrary.hpp"

#include <array>
#include <cassert>
#include <vector>

#include "graph/ResourceState.hpp"
#include "spdlog/spdlog.h"

namespace brassica::render {

	std::span<const BlendAttachmentConfig> BlendConfigFor(BlendMode mode) {
		static constexpr std::array<BlendAttachmentConfig, 1> kOpaque{BlendAttachmentConfig{.blendEnable = false}};
		static constexpr std::array<BlendAttachmentConfig, 1> kAlphaBlend{BlendAttachmentConfig{
			.blendEnable = true,
			.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
			.colorBlendOp = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eOne,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp = vk::BlendOp::eAdd,
		}};
		// McGuire weighted-blended OIT: attachment 0 is the premultiplied-by-weight color
		// accumulator (additive), attachment 1 is the revealage accumulator (tracks the
		// product of (1 - weight) across every fragment, so a resolve pass can recover
		// averaged color as accum / max(1 - revealage, epsilon)).
		static constexpr std::array<BlendAttachmentConfig, 2> kWeightedOit{
			BlendAttachmentConfig{
				.blendEnable = true,
				.srcColorBlendFactor = vk::BlendFactor::eOne,
				.dstColorBlendFactor = vk::BlendFactor::eOne,
				.colorBlendOp = vk::BlendOp::eAdd,
				.srcAlphaBlendFactor = vk::BlendFactor::eOne,
				.dstAlphaBlendFactor = vk::BlendFactor::eOne,
				.alphaBlendOp = vk::BlendOp::eAdd,
			},
			BlendAttachmentConfig{
				.blendEnable = true,
				.srcColorBlendFactor = vk::BlendFactor::eZero,
				.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor,
				.colorBlendOp = vk::BlendOp::eAdd,
				.srcAlphaBlendFactor = vk::BlendFactor::eZero,
				.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
				.alphaBlendOp = vk::BlendOp::eAdd,
			},
		};

		switch (mode) {
		case BlendMode::Opaque:
			return kOpaque;
		case BlendMode::AlphaBlend:
			return kAlphaBlend;
		case BlendMode::WeightedOit:
			return kWeightedOit;
		}
		return kOpaque;
	}

	const StencilConfig& StencilConfigFor(StencilMode mode) {
		static constexpr StencilConfig kDisabled{};
		static constexpr StencilConfig kWrite{
			.failOp = vk::StencilOp::eKeep,
			.passOp = vk::StencilOp::eReplace,
			.depthFailOp = vk::StencilOp::eKeep,
			.compareOp = vk::CompareOp::eAlways,
			.compareMask = 0xFF,
			.writeMask = 0xFF,
		};
		static constexpr StencilConfig kTestEqual{
			.failOp = vk::StencilOp::eKeep,
			.passOp = vk::StencilOp::eKeep,
			.depthFailOp = vk::StencilOp::eKeep,
			.compareOp = vk::CompareOp::eEqual,
			.compareMask = 0xFF,
			.writeMask = 0, // never writes -- belt-and-suspenders alongside passOp/failOp=eKeep
		};
		static constexpr StencilConfig kTestNotEqual{
			.failOp = vk::StencilOp::eKeep,
			.passOp = vk::StencilOp::eKeep,
			.depthFailOp = vk::StencilOp::eKeep,
			.compareOp = vk::CompareOp::eNotEqual,
			.compareMask = 0xFF,
			.writeMask = 0,
		};

		switch (mode) {
		case StencilMode::Disabled:
			return kDisabled;
		case StencilMode::Write:
			return kWrite;
		case StencilMode::TestEqual:
			return kTestEqual;
		case StencilMode::TestNotEqual:
			return kTestNotEqual;
		}
		return kDisabled;
	}

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
		key.words.push_back(static_cast<std::uint64_t>(request.state.blendMode));
		key.words.push_back(static_cast<std::uint64_t>(request.state.stencilMode));
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
		bool hasMeshOrTask = false;
		for (GraphicsShader* shader : request.stages) {
			if (shader) {
				stages.push_back(shader->GetStageCreateInfo());
				vk::ShaderStageFlagBits stageFlag = shader->GetStageFlag();
				if (stageFlag == vk::ShaderStageFlagBits::eMeshEXT || stageFlag == vk::ShaderStageFlagBits::eTaskEXT) {
					hasMeshOrTask = true;
				}
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

		std::span<const BlendAttachmentConfig> blendConfig = BlendConfigFor(request.state.blendMode);
		// A multi-entry mode (WeightedOit) describes one config per color attachment, in
		// order -- it must be used with exactly that many attachments. A single-entry mode
		// broadcasts to however many attachments the pipeline declares. Any other arrangement
		// is a caller bug (a mode built for N attachments used with the wrong pipeline shape),
		// so this asserts rather than silently falling back to attachment 0 for every slot.
		assert(
			(blendConfig.size() == 1 || blendConfig.size() == request.colorFormats.size()) &&
			"BlendMode's attachment count must be 1 (broadcast) or match colorFormats exactly"
		);

		std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments(request.colorFormats.size());
		for (std::size_t i = 0; i < colorBlendAttachments.size(); ++i) {
			const BlendAttachmentConfig& cfg = blendConfig[blendConfig.size() == 1 ? 0 : i];
			vk::PipelineColorBlendAttachmentState& attachment = colorBlendAttachments[i];
			attachment.setColorWriteMask(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
				vk::ColorComponentFlagBits::eA
			);
			attachment.setBlendEnable(cfg.blendEnable ? VK_TRUE : VK_FALSE);
			attachment.setSrcColorBlendFactor(cfg.srcColorBlendFactor);
			attachment.setDstColorBlendFactor(cfg.dstColorBlendFactor);
			attachment.setColorBlendOp(cfg.colorBlendOp);
			attachment.setSrcAlphaBlendFactor(cfg.srcAlphaBlendFactor);
			attachment.setDstAlphaBlendFactor(cfg.dstAlphaBlendFactor);
			attachment.setAlphaBlendOp(cfg.alphaBlendOp);
		}

		vk::PipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.setLogicOpEnable(VK_FALSE);
		colorBlending.setAttachments(colorBlendAttachments);

		const StencilConfig& stencilConfig = StencilConfigFor(request.state.stencilMode);
		vk::StencilOpState   stencilOpState{
			stencilConfig.failOp,
			stencilConfig.passOp,
			stencilConfig.depthFailOp,
			stencilConfig.compareOp,
			stencilConfig.compareMask,
			stencilConfig.writeMask,
			0, // reference -- dynamic state (eStencilReference), set per-draw by BindGraphics
        };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.setDepthTestEnable(request.state.depthTest ? VK_TRUE : VK_FALSE);
		depthStencil.setDepthWriteEnable(request.state.depthWrite ? VK_TRUE : VK_FALSE);
		depthStencil.setDepthCompareOp(request.state.depthCompareOp);
		depthStencil.setDepthBoundsTestEnable(VK_FALSE);
		depthStencil.setStencilTestEnable(request.state.stencilMode != StencilMode::Disabled ? VK_TRUE : VK_FALSE);
		depthStencil.setFront(stencilOpState);
		depthStencil.setBack(stencilOpState);

		std::array<vk::DynamicState, 3> dynamicStates{
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor,
			vk::DynamicState::eStencilReference,
		};
		vk::PipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.setDynamicStates(dynamicStates);

		// Attachment-only VRS: only attachmentFragmentShadingRate is enabled (Engine.cpp), not
		// primitiveFragmentShadingRate/fragmentShadingRateNonTrivialCombinerOps, so both combiner
		// ops must be "trivial" (KEEP or REPLACE only -- MAX/MIN/etc. need the nontrivial-combiner
		// feature this plan doesn't request). combinerOps[0] combines the pipeline's own fixed
		// {1,1} fragmentSize with the primitive rate; KEEP here is a no-op since nothing can ever
		// write a primitive rate without primitiveFragmentShadingRate enabled, so the primitive
		// input is always {1,1} too. combinerOps[1] combines that intermediate with the
		// attachment's rate; REPLACE makes the attachment's rate win unconditionally -- exactly
		// what "derive this frame's shading rate from a compute-authored image" needs.
		VkPipelineFragmentShadingRateStateCreateInfoKHR shadingRateState{};
		shadingRateState.sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
		shadingRateState.fragmentSize = {1, 1};
		shadingRateState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
		shadingRateState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;

		vk::PipelineRenderingCreateInfo renderingCreateInfo{};
		if (!request.colorFormats.empty()) {
			renderingCreateInfo.setColorAttachmentFormats(request.colorFormats);
		}
		if (request.depthFormat != vk::Format::eUndefined) {
			renderingCreateInfo.setDepthAttachmentFormat(request.depthFormat);
			if (graph::HasStencilComponent(request.depthFormat)) {
				renderingCreateInfo.setStencilAttachmentFormat(request.depthFormat);
			}
		}
		if (request.state.enableShadingRate) {
			renderingCreateInfo.setPNext(&shadingRateState);
		}

		vk::GraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.setPNext(&renderingCreateInfo);
		// Required whenever this pipeline is used with an attachment-rate VRS image bound via
		// dynamic rendering's pNext (DynamicRenderingWrapper::Begin, PhysicalExecutionBackend.hpp)
		// -- VUID-VkGraphicsPipelineCreateInfo-renderPass-06576-adjacent: a pipeline built without
		// this flag is undefined behavior if later bound during a render pass instance that has a
		// shading-rate attachment.
		if (request.state.enableShadingRate) {
			pipelineInfo.setFlags(vk::PipelineCreateFlagBits::eRenderingFragmentShadingRateAttachmentKHR);
		}
		pipelineInfo.setStages(stages);
		pipelineInfo.setPVertexInputState(hasMeshOrTask ? nullptr : &vertexInputInfo);
		pipelineInfo.setPInputAssemblyState(hasMeshOrTask ? nullptr : &inputAssembly);
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
