#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "passes/TerrainBiomeNode.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

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
		render::PipelineLibrary*    pipelineLibrary = nullptr;
		ComputeShader               genShader;
		CloudShadowGenPushConstants push{};
		glm::vec3                   cameraPos{0.0f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/cloud_shadow_gen.comp")) {
				spdlog::critical("CloudShadowGenNode shader compilation failed.");
				throw std::runtime_error("CloudShadowGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<CloudShadowMap>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&genShader); }

		void Destroy(vk::Device device) { genShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			cameraPos = p.cameraPosition;
		}

		// S4: no kPhase -- Read<TerrainWeatherBiomeTexture>() restored (catalog-converted since
		// batch 5) orders it after TerrainBiomeNode's sole Modify, and Read<CloudVolumeCascade*>()
		// orders it after CloudVolumeGenNode's sole Create of each, both via ordinary resource
		// edges, so the SubPhase::LightPreparation placement was redundant.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<CloudShadowMap>();
			r.Read<CloudVolumeCascade0>();
			r.Read<CloudVolumeCascade1>();
			r.Read<CloudVolumeCascade2>();
			r.Read<TerrainWeatherBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.cascadeSampledIdx.x = ctx.Index<CloudVolumeCascade0>();
			push.cascadeSampledIdx.y = ctx.Index<CloudVolumeCascade1>();
			push.cascadeSampledIdx.z = ctx.Index<CloudVolumeCascade2>();
			push.cascadeSampledIdx.w = ctx.Index<TerrainWeatherBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.shadowMapStorageIdx = ctx.StorageIndex<CloudShadowMap>();
			push.cameraPos = glm::vec4(cameraPos, 320000.0f);

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Dispatch 1024 / 16 = 64 workgroups per dimension
			vkCmd.dispatch(64, 64, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudShadowGenNode);

} // namespace brassica
