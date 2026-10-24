#pragma once

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
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct TerrainShadowPushConstants {
		alignas(16) glm::vec4 cameraPos{0.0f};          // xyz = cameraPos, w = time
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 1.0f}; // xyz = sunDir, w = sunIntensity
		alignas(16) glm::uvec4 gridParams{0};           // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = mapDim
		alignas(16) glm::uvec4 indices{0xFFFFFFFFu};     // x = minMaxIndex, y = clipmapIndex, z = shadowMapStorageIdx, w = cubemapMinMaxIndex
		alignas(16) glm::uvec4 cascadeLODs{1, 3, 6, 0}; // x = cascade 0 LOD (1), y = cascade 1 LOD (3), z = cascade 2 LOD (6)
		alignas(16) glm::uvec4 cubemapIndices{0xFFFFFFFFu}; // x = cubemapIndex, y = unused, z = unused, w = unused
	};

	static_assert(
		sizeof(TerrainShadowPushConstants) == 96,
		"TerrainShadowPushConstants size must be 96 bytes"
	);

	// Generates a 3-layer 2D shadow map for the sun using minmax mipmap traversal.
	struct TerrainShadowNode: render::NodeRegistrar<TerrainShadowNode> {
		render::PipelineLibrary*   pipelineLibrary = nullptr;
		ComputeShader              genShader;
		TerrainShadowPushConstants push{};
		glm::vec3                  cameraPos{0.0f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_shadow.comp")) {
				spdlog::critical("TerrainShadowNode shader compilation failed.");
				throw std::runtime_error("TerrainShadowNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<TerrainShadowMap>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&genShader); }

		void Destroy(vk::Device device) { genShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.cameraPos = glm::vec4(cameraPos, p.time);
			push.sunDir = glm::vec4(p.sunDir, (p.sunRadiance.r > 0.0f || p.sunRadiance.g > 0.0f || p.sunRadiance.b > 0.0f) ? 1.0f : 0.0f);
			push.gridParams = p.terrainGridParams;
			push.cascadeLODs = glm::uvec4(1u, 3u, 6u, 0u);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainShadowMap>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainCubemapTexture>();
			r.Read<TerrainCubemapMinMaxTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.indices.x = ctx.Index<TerrainMinMaxTexture>();
			push.indices.y = ctx.Index<TerrainClipmapTexture>();
			push.indices.z = ctx.StorageIndex<TerrainShadowMap>();
			push.indices.w = ctx.Index<TerrainCubemapMinMaxTexture>();
			push.cubemapIndices.x = ctx.Index<TerrainCubemapTexture>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Dispatch 512 / 16 = 32 workgroups in x and y (evaluating 1/4 of 1024x1024 pixels per frame), 3 in z
			vkCmd.dispatch(32, 32, 3);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainShadowNode);

} // namespace brassica
