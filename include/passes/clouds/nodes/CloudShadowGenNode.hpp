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
#include "passes/biome/resources.hpp"
#include "passes/clouds/resources.hpp"
#include "passes/biome/nodes/TerrainBiomeNode.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct CloudShadowGenPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::uvec4 volumeIndices{0};
		alignas(16) glm::vec4 cameraPos{0.0f};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t shadowMapStorageIdx{0};
	};

	// Evaluates top-down atmospheric and cloud extinction, storing 5 layers per texel along the
	// sun ray from where it enters the cloud shell down to the real terrain/floor intersection
	// (found by traceTerrainOcclusion, not a fixed altitude):
	// Layer 0: Metadata (cloud entry height, terrain intersection height)
	// Layer 1-3: 25%/50%/75% of the way from cloud entry to the terrain intersection
	// Layer 4: The terrain intersection itself (totals for the full ray)
	// Each layer stores max extinction, average extinction, optical depth, and transmittance.
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
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&genShader); }

		void Destroy(vk::Device device) { genShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			cameraPos = p.cameraPosition;
		}

		// S4: no kPhase -- Read<TerrainWeatherBiomeTexture>() restored (catalog-converted since
		// batch 5) orders it after TerrainBiomeNode's sole Modify, and Read<CloudDetailVolume>()
		// orders it after CloudVolumeGenNode's sole Create, both via ordinary resource edges, so
		// the SubPhase::LightPreparation placement was redundant.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<CloudShadowMap>();
			r.Read<CloudDetailVolume>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Read<TerrainMinMaxTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.volumeIndices.x = ctx.Index<CloudDetailVolume>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
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
