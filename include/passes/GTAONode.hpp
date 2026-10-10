#pragma once

#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "lighting/ILightManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	struct GTAOPushConstants {
		std::uint32_t gPositionIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t outGtaoIndex{0};
		glm::uvec2    screenSize{0, 0};
	};

	struct GTAONode: render::NodeRegistrar<GTAONode> {
		using Resources = graph::Declares<
			graph::Read<GBufferPosition>,
			graph::Read<GBufferNormal>,
			graph::Read<GBufferDepth>,
			graph::Create<GTAOTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::GIComput;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            gtaoShader;
		GTAOPushConstants        push{};
		std::uint32_t            width{0};
		std::uint32_t            height{0};
		bool                     enabled{true};

		void SetFrameParams(const render::NodeFrameParams& p) {
			if (ServiceLocator::Instance().Has<ILightManager>()) {
				enabled = ServiceLocator::Instance().Get<ILightManager>()->IsGTAOEnabled();
			}
		}

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!gtaoShader.CompileComputeFromFile(services.device, "shaders/gtao.comp")) {
				spdlog::critical("GTAONode shader compilation failed.");
				throw std::runtime_error("GTAONode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&gtaoShader);
			}
		}

		void Destroy(vk::Device device) {
			gtaoShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = enabled};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(width, height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(width, height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Read,
					.desc = graph::DepthBufferDesc(width, height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GTAOTexture>(),
					.access = graph::AccessKind::Write,
					.desc = GTAOTextureDesc(width, height),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			if (width == 0 || height == 0) {
				return;
			}

			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.outGtaoIndex = ctx.StorageIndex<GTAOTexture>();
			push.screenSize = glm::uvec2(width, height);

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(GTAOPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &gtaoShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(GTAOPushConstants),
				&push
			);

			std::uint32_t dispatchX = (width + 7) / 8;
			std::uint32_t dispatchY = (height + 7) / 8;
			vkCmd.dispatch(dispatchX, dispatchY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(GTAONode);

} // namespace brassica
