#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/terrain/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "passes/terrain/TerrainAccelerationStructure.hpp"
#include "passes/terrain/TerrainManager.hpp"

namespace brassica {

	struct TerrainGenPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		std::uint32_t clipmapStorageIdx{0};
		std::uint32_t minMaxStorageIdx{0};
		std::uint32_t biomeStorageIdx{0};
		bool          forceRegeneration = true;
	};

	// S4: no kPhase -- the old comment here ("the graph validator rejects a later-phase producer
	// satisfying an earlier-phase consumer's input") described a hard phase-partition check that
	// no longer exists (removed at S3, see Graph.hpp's Compile() comment). This node is the sole
	// Create<> of TerrainClipmapTexture/TerrainMinMaxTexture/TerrainBiomeTexture, so
	// Create-before-accessor already orders it before every reader -- including
	// ParticleSystemNode's Prepare-phase nodes -- as a real resource edge, regardless of what
	// phase (if any) either side is on.
	struct TerrainGenNode: render::NodeRegistrar<TerrainGenNode> {
		render::PipelineLibrary*         pipelineLibrary = nullptr;
		graph::PhysicalResourceRegistry* physicalRegistry = nullptr;
		ComputeShader                    genShader;
		ComputeShader                    aabbShader;
		TerrainAccelerationStructure*    terrainAS = nullptr;
		TerrainGenPushConstants          push{};
		glm::vec3                        cameraPos{0.0f};
		bool                             hasUpdate{true};
		bool                             forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			terrainAS = services.terrainAS;
			physicalRegistry = services.physicalRegistry;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/terrain_gen.comp") ||
			    !aabbShader.CompileComputeFromFile(services.device, "shaders/terrain_aabb.comp")) {
				spdlog::critical("TerrainGenNode shader compilation failed.");
				throw std::runtime_error("TerrainGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
			watcher.RegisterShader(&aabbShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			aabbShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			push.gridParams = p.terrainGridParams;
			hasUpdate = p.cameraPosition != p.previousCameraPosition;
			forceRegeneration = p.forceRegeneration;

			// Publishes this frame's numLODs onto all three keys' catalog instances -- the one
			// piece of runtime config their Desc() needs beyond FrameContext (ResourceKeys.hpp).
			// Idempotent and cheap to call every frame (Get<K> only default-constructs once).
			if (physicalRegistry) {
				auto& catalog = physicalRegistry->Catalog();
				catalog.Get<TerrainClipmapTexture>().numLODs = p.terrainGridParams.x;
				catalog.Get<TerrainMinMaxTexture>().numLODs = p.terrainGridParams.x;
				catalog.Get<TerrainBiomeTexture>().numLODs = p.terrainGridParams.x;
			}
		}

		// These three are genuinely Create<>, not Modify<>, despite this node running every
		// frame: they're Lifetime::Transient keys with desc.persistent=true (TerrainLODImage2D,
		// resources/ResourceTypes.hpp) -- Lifetime::Persistent would also skip RelationalCompiler's
		// Create-before-accessor ordering rule, which silently broke scheduling against
		// TerrainMinMaxDownsampleNode's same-frame Modify (caught by the golden-fixture test: see
		// its "ambiguous write order" failure). desc.persistent=true gets the narrower effect
		// actually wanted here -- alias-pool bypass + zero-init-clear exemption only, nothing
		// about scheduling -- so "recreating" them each frame is really just re-asserting the same
		// desc through ProvisionTexture's normal desc-match early return, not destroying state.
		// TerrainTLAS needs no .Desc() either -- Provision()'s AccelerationStructure detection now
		// checks the registry's own m_accelStructs map directly instead of a realization's raw
		// desc.kind (PhysicalRegistry.hpp).
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<TerrainClipmapTexture>();
			r.Create<TerrainMinMaxTexture>();
			r.Create<TerrainBiomeTexture>();
			r.Modify<TerrainTLAS>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (!hasUpdate && !forceRegeneration) {
				return;
			}

			push.clipmapStorageIdx = ctx.StorageIndex<TerrainClipmapTexture>();
			push.minMaxStorageIdx = ctx.StorageIndex<TerrainMinMaxTexture>();
			push.biomeStorageIdx = ctx.StorageIndex<TerrainBiomeTexture>();
			push.forceRegeneration = forceRegeneration;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			uint32_t          groupX = (push.gridParams.w + 15) / 16;
			uint32_t          groupY = (push.gridParams.w + 15) / 16;
			uint32_t          groupZ = push.gridParams.x;
			vkCmd.dispatch(groupX, groupY, groupZ);
		}
	};

	BRASSICA_REGISTER_NODE(TerrainGenNode);

} // namespace brassica
