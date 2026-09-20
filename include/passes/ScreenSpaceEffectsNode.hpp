#pragma once

#include <array>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/ScreenSpacePushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	struct ScreenSpaceEffectsNode: render::NodeRegistrar<ScreenSpaceEffectsNode> {
		static constexpr graph::Phase kPhase = SubPhase::GIComput;

		using Resources = graph::Declares<
			GBuffer<graph::Read>,
			graph::Create<ScreenSpaceIndirectAO>,
			graph::Create<ScreenSpaceShadow>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            shader;
		ScreenSpacePushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			shader.CompileComputeFromFile(services.device, "shaders/effects/screen_space_effects.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&shader); }

		void Destroy(vk::Device device) { shader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = true};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ScreenSpaceIndirectAO>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ScreenSpaceShadow>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.outIndirectAOIndex = ctx.StorageIndex<ScreenSpaceIndirectAO>();
			push.outShadowMaskIndex = ctx.StorageIndex<ScreenSpaceShadow>();

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ScreenSpacePushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(ScreenSpacePushConstants),
				&push
			);

			vkCmd.dispatch((ctx.width + 7) / 8, (ctx.height + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ScreenSpaceEffectsNode);

} // namespace brassica
