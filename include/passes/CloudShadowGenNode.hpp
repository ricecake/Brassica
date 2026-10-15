#pragma once

#include <array>
#include <cstdint>

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
#include "spdlog/spdlog.h"

namespace brassica {

	struct CloudShadowGenPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::uvec4 cascadeSampledIdx{0};
		alignas(16) glm::vec4 cameraPos{0.0f};
		alignas(16) glm::vec4 fogParams{0.003f, 0.003f, 0.0f, 0.6f};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t shadowMapStorageIdx{0};
		std::uint32_t _pad[2]{0};
	};

	// Evaluates top-down atmospheric and cloud extinction across 5 altitude layers:
	// Layer 0: Top of cloud layer (20,000m)
	// Layer 1: Middle of cloud layer (12,500m)
	// Layer 2: Bottom of cloud layer (5,000m)
	// Layer 3: Middle of sub-cloud volume (2,500m)
	// Layer 4: Ground level / totals (0m)
	// Stores max extinction, average extinction, optical depth, and transmittance.
	struct CloudShadowGenNode: render::NodeRegistrar<CloudShadowGenNode> {
		using Resources = graph::Declares<
			graph::Read<CloudVolumeCascade0>,
			graph::Read<CloudVolumeCascade1>,
			graph::Read<CloudVolumeCascade2>,
			graph::Read<TerrainWeatherBiomeTexture>,
			graph::Create<CloudShadowMap>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary*   pipelineLibrary = nullptr;
		ComputeShader              genShader;
		CloudShadowGenPushConstants push{};
		glm::vec3                  cameraPos{0.0f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/cloud_shadow_gen.comp")) {
				spdlog::critical("CloudShadowGenNode shader compilation failed.");
				throw std::runtime_error("CloudShadowGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			cameraPos = p.cameraPosition;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(1);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudShadowMap>(),
					.access = graph::AccessKind::Write,
					.desc = CloudShadowMapDesc(1024, 1024),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout))
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet))
			};

			push.cascadeSampledIdx.x = ctx.Index<CloudVolumeCascade0>();
			push.cascadeSampledIdx.y = ctx.Index<CloudVolumeCascade1>();
			push.cascadeSampledIdx.z = ctx.Index<CloudVolumeCascade2>();
			push.cascadeSampledIdx.w = ctx.Index<TerrainWeatherBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.shadowMapStorageIdx = ctx.StorageIndex<CloudShadowMap>();
			push.cameraPos = glm::vec4(cameraPos, 320000.0f);

			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudShadowGenPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &genShader,
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
				sizeof(CloudShadowGenPushConstants),
				&push
			);

			// Dispatch 1024 / 16 = 64 workgroups per dimension
			vkCmd.dispatch(64, 64, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudShadowGenNode);

} // namespace brassica
