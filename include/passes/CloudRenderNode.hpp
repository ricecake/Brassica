#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct alignas(16) CloudRenderPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f};
		alignas(16) glm::uvec4 cascadeSampledIdx{0};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t transmittanceIndex{0};

		std::uint32_t skyViewIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t cloudShadowIndex{0};
		float         cirrusOpacity{0.0125f};

		std::uint32_t outPackedColorStorageIdx{0};
		std::uint32_t outPackedDepthStorageIdx{0};
		std::uint32_t outPackedVelocityStorageIdx{0};
		std::uint32_t frameIndex{0};

		std::uint32_t fullWidth{0};
		std::uint32_t fullHeight{0};
	};

	// Compute pass that evaluates 1/16th of threads each frame into full-resolution storage images for low-res raymarching.
	struct CloudRenderNode: render::NodeRegistrar<CloudRenderNode> {
		using Resources = graph::Declares<
			GBuffer<graph::Read>,
			graph::Read<CloudVolumeCascade0>,
			graph::Read<CloudVolumeCascade1>,
			graph::Read<CloudVolumeCascade2>,
			graph::Read<TransmittanceLUT>,
			graph::Read<SkyViewLUT>,
			graph::Read<TerrainWeatherBiomeTexture>,
			graph::Read<CloudShadowMap>,
			graph::Create<CloudRenderTexture>,
			graph::Create<CloudDepthTexture>,
			graph::Create<CloudVelocityTexture>>;

		static constexpr graph::Phase kPhase = graph::Phase(1250);

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader; // shaders/cloud_render.comp
		CloudRenderPushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/cloud_render.comp")) {
				spdlog::critical("CloudRenderNode shader compilation failed.");
				throw std::runtime_error("CloudRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			push.sunRadianceAndSkyExp = glm::vec4(p.sunRadiance, p.skyExposure);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(9);

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudRenderTexture>(),
					.access = graph::AccessKind::Write,
					.desc = CloudRenderTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudDepthTexture>(),
					.access = graph::AccessKind::Write,
					.desc = CloudDepthTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVelocityTexture>(),
					.access = graph::AccessKind::Write,
					.desc = CloudVelocityTextureDesc(ctx.width, ctx.height),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade0>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade1>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade2>(),
					.access = graph::AccessKind::Read,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<SkyViewLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudShadowMap>(),
					.access = graph::AccessKind::Read,
					.desc = CloudShadowMapDesc(1024, 1024),
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.cascadeSampledIdx.x = ctx.Index<CloudVolumeCascade0>();
			push.cascadeSampledIdx.y = ctx.Index<CloudVolumeCascade1>();
			push.cascadeSampledIdx.z = ctx.Index<CloudVolumeCascade2>();
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.cloudShadowIndex = ctx.Index<CloudShadowMap>();

			push.outPackedColorStorageIdx = ctx.StorageIndex<CloudRenderTexture>();
			push.outPackedDepthStorageIdx = ctx.StorageIndex<CloudDepthTexture>();
			push.outPackedVelocityStorageIdx = ctx.StorageIndex<CloudVelocityTexture>();
			push.frameIndex = static_cast<std::uint32_t>(ctx.frameIndex);
			push.fullWidth = ctx.width;
			push.fullHeight = ctx.height;

			std::array<ComputeShader*, 1>          stages{&compShader};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudRenderPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
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
				sizeof(CloudRenderPushConstants),
				&push
			);

			std::uint32_t groupX = (ctx.width + 7) / 8;
			std::uint32_t groupY = (ctx.height + 7) / 8;
			vkCmd.dispatch(groupX, groupY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudRenderNode);

} // namespace brassica
