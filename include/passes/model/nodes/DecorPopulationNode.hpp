#pragma once

#include <cstdint>
#include <vector>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/DecorChunkAllocator.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/biome/resources.hpp"
#include "passes/model/ProceduralArchetypeCatalog.hpp"
#include "passes/model/resources.hpp"
#include "passes/terrain/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/CameraData.hpp"

namespace brassica {

	// Forward-declared, not #included -- TypeName<X>() (used by After<X>() below) only needs a
	// declaration, same pattern as ParticleSystemNode.hpp's own After<WaterNode>()/Before<WaterNode>().
	struct ModelInstanceScatterNode;

	struct DecorGenPushConstants {
		float         worldOffsetX{0.0f};
		float         worldOffsetZ{0.0f};
		std::uint32_t cellSeed{0};
		std::uint32_t blockBaseSlot{0};
		std::uint32_t instanceBufIdx{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t terrainTextureDim{2048u};
		float         tileSize{16.0f};
		std::uint32_t treeModelBase{0};
		std::uint32_t bushModelBase{0};
		std::uint32_t rockModelBase{0};
		float         treeRatio{0.1f};
		float         bushRatio{0.1f};
		float         rockRatio{0.05f};
		float         density{0.65f};
	};

	static_assert(sizeof(DecorGenPushConstants) == 64, "DecorGenPushConstants size must be 64 bytes");

	struct DecorClearPushConstants {
		std::uint32_t blockBaseSlot{0};
		std::uint32_t instanceBufIdx{0};
	};

	static_assert(sizeof(DecorClearPushConstants) == 8, "DecorClearPushConstants size must be 8 bytes");

	// Stage 5.2 of the GPU-driven model-instance plan: the persistent, disocclusion-driven decor
	// population that replaces FoliageNode's old per-frame tree/bush/rock emission. CPU side is
	// purely allocation bookkeeping (DecorChunkAllocator); the actual placement decision (which
	// archetype, where, what seed) happens on the GPU in decor_gen_chunk.comp, writing directly
	// into ModelInstanceBuffer's reserved decor range -- matching the plan's own "GPU-side
	// placement, not a CPU loop." ModelCullNode/ModelRenderNode need zero changes: they just see
	// more populated slots in the same buffer they already walk.
	//
	// Tile size / texture dim use FoliagePushConstants' own *default* values (16.0f / 2048) --
	// a first-pass simplification, not wired through the same dynamic NodeFrameParams path
	// FoliageNode's own gridParams.w uses. Revisit if those are ever tuned away from their
	// defaults in practice.
	struct DecorPopulationNode: render::NodeRegistrar<DecorPopulationNode> {
		render::PipelineLibrary*  pipelineLibrary = nullptr;
		ComputeShader             genShader;
		ComputeShader             clearShader;
		DecorChunkAllocator       allocator;
		std::uint32_t             terrainTextureDim{2048u};

		void SetFrameParams(const render::NodeFrameParams& p) {
			terrainTextureDim = p.terrainGridParams.w;
		}

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/decor_gen_chunk.comp") ||
			    !clearShader.CompileComputeFromFile(services.device, "shaders/decor_clear_block.comp")) {
				spdlog::critical("DecorPopulationNode shader compilation failed.");
				throw std::runtime_error("DecorPopulationNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&genShader);
				services.shaderWatcher->RegisterShader(&clearShader);
			}
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
			clearShader.Destroy(device);
		}

		[[nodiscard]] static ModelRegistry* QueryRegistry() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
				return ServiceLocator::Instance().Get<ModelRegistry>().get();
			}
			return nullptr;
		}

		// Engine.cpp provides this once at startup from RegisterProceduralArchetypes()'s return
		// value (same ServiceLocator::Provide<>() pattern as ModelRegistry/ModelInstanceManager
		// above) -- not a setter reached through a node-instance lookup, since no such lookup
		// exists for registered nodes.
		[[nodiscard]] static ProceduralArchetypeIndices* QueryArchetypeIndices() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ProceduralArchetypeIndices>()) {
				return ServiceLocator::Instance().Get<ProceduralArchetypeIndices>().get();
			}
			return nullptr;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = QueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(totalMeshletCount > 0 && QueryArchetypeIndices() != nullptr);
			// ModelInstanceScatterNode's Create<>() does ModelInstanceBuffer's one-time zero-clear
			// on the first frame it's ever provisioned -- without a real edge forcing it before
			// this node's own writes, the two are "two writers, nothing ordering them," a hard
			// ambiguity error (RelationalCompiler.hpp), and AnyOrder() would be actively wrong here
			// (its deterministic tie-break is alphabetical by node name, which picks
			// DecorPopulationNode before ModelInstanceScatterNode -- the clear would run after this
			// node's writes and erase them on that first frame). After<>() is a real scheduler edge,
			// not the design-doc's inert §C/§E shim its own comment describes -- confirmed by
			// reading RelationalCompiler.hpp directly, not assumed from that stale comment.
			r.After<ModelInstanceScatterNode>();
			r.Modify<ModelInstanceBuffer>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainWeatherBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg || !physReg->GetBuffer<ModelInstanceBuffer>()) {
				// Same "not provisioned until first-ever ModelInstanceScatterNode Create<>()"
				// guard every other ModelInstanceBuffer consumer uses.
				return;
			}

			auto* archetypeIndices = QueryArchetypeIndices();
			if (!archetypeIndices) {
				// Setup() already gates Active() on this, but a null check here costs nothing
				// and matches ModelInstanceScatterNode's own belt-and-suspenders style.
				return;
			}

			glm::vec3 cameraPosition(0.0f);
			if (ServiceLocator::Instance().Has<CameraData>()) {
				cameraPosition = ServiceLocator::Instance().Get<CameraData>()->position;
			}

			constexpr float kTileSize = 16.0f; // FoliagePushConstants' own default baseTileSize
			std::vector<std::uint32_t> freedBlockBases;
			std::vector<DecorGenerateEntry> toGenerate = allocator.UpdateAllocation(cameraPosition, kTileSize, freedBlockBases);

			std::uint32_t instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			std::uint32_t clipmapIdx = ctx.Index<TerrainClipmapTexture>();
			std::uint32_t weatherBiomeIdx = ctx.Index<TerrainWeatherBiomeTexture>();
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			for (std::uint32_t blockBase : freedBlockBases) {
				DecorClearPushConstants push{};
				push.blockBaseSlot = blockBase;
				push.instanceBufIdx = instanceBufIdx;
				render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, clearShader, push);
				if (!resolved.pipeline) {
					continue;
				}
				vkCmd.dispatch(1, 1, 1);
			}

			for (const DecorGenerateEntry& entry : toGenerate) {
				DecorGenPushConstants push{};
				push.worldOffsetX = entry.worldOffset.x;
				push.worldOffsetZ = entry.worldOffset.z;
				push.cellSeed = entry.cellSeed;
				push.blockBaseSlot = entry.blockBaseSlot;
				push.instanceBufIdx = instanceBufIdx;
				push.clipmapIndex = clipmapIdx;
				push.weatherBiomeIndex = weatherBiomeIdx;
				push.terrainTextureDim = terrainTextureDim;
				push.tileSize = kTileSize;
				push.treeModelBase = archetypeIndices->treeBase;
				push.bushModelBase = archetypeIndices->bushBase;
				push.rockModelBase = archetypeIndices->rockBase;

				render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
				if (!resolved.pipeline) {
					continue;
				}
				vkCmd.dispatch(1, 1, 1);
			}
		}
	};

	BRASSICA_REGISTER_NODE(DecorPopulationNode);

} // namespace brassica
