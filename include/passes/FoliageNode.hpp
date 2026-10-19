#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

#include "constants.h"
#include "foliage/IFoliageManager.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct FoliagePushConstants {
		glm::uvec4    gridParams{8u, 16u, 2048u, 1024u}; // x = numLODs, y = tilesPerRow, z = totalTiles, w = textureDim
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		// Reserved: TerrainBiomeTexture's per-location value is hardcoded to 0 in
		// terrain_gen.comp. The real per-location biome signal is weatherBiomeIndex below, via
		// evaluateWhittakerBiome (helpers/whittaker.glsl) -> biomeTableIndex's per-biome row.
		std::uint32_t biomeIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		float         windTime{0.0f};
		float         lodBaseRange{20.0f};
		float         lodScaleFactor{2.0f};
		float         baseTileSize{16.0f};
		float         flowerRatio{0.15f};
		float         fernRatio{0.20f};
		float         rockRatio{0.05f};
		float         seaweedRatio{0.05f};
		float         bushRatio{0.10f};
		float         treeRatio{0.10f};
		float         lengthMultiplier{1.0f};
		float         widthMultiplier{1.0f};
		float         densityMultiplier{1.0f};
		float         windMultiplier{1.0f};
		float         rigidityMultiplier{1.0f};
		// Appended, not interleaved -- see feedback_push_constant_struct_editing. These replace
		// what used to be padding0/padding1 (same offsets, same size) plus a real append after.
		std::uint32_t biomeTableIndex{0};
		float         baseScale{0.5f};
		float         waterLevel{0.0f};
		std::uint32_t terrainNumLevels{8};
		std::uint32_t foliageFlags{0}; // bit 0: terrain occlusion culling
		std::uint32_t hizIndex{0};     // bindless index for History<HiZTexture>
	};

	static_assert(sizeof(FoliagePushConstants) == 116, "FoliagePushConstants size must be 116 bytes");

	// S4: no kPhase -- see EntityNode.hpp's identical reasoning (TerrainNode's Create-before-
	// accessor orders this node after it, and every reader of the shared GBuffer keys gets a
	// real default edge from this node's writes regardless of phase).
	struct FoliageNode: render::NodeRegistrar<FoliageNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			// Engine convention is standard Z (0=near, 1=far; see Engine.cpp's plain
		    // glm::perspective, no reverse-Z remap), matching TerrainNode/WaterNode/EntityNode's
		    // eLess (and GraphicsPipelineState's own default) -- eGreaterOrEqual here was backwards,
		    // so foliage only ever drew where it was farther than or equal to whatever terrain had
		    // already written, i.e. almost nowhere a normal above-ground camera could see it.
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*               pipelineLibrary = nullptr;
		TaskShader                             taskShader;
		MeshShader                             meshShader;
		FragmentShader                         fragShader;
		const brassica::DispatchLoaderDynamic* dls = nullptr;
		FoliagePushConstants                   push{};
		bool                                   enabled{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/foliage.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/foliage.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/foliage.frag")) {
				spdlog::critical("FoliageNode shader compilation failed.");
				throw std::runtime_error("FoliageNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.windTime = p.time;
			// gridParams.w previously stayed at its hardcoded default (1088) forever -- never
			// synced to the real terrain texture size, so foliage sampled slightly wrong texels.
			push.gridParams.w = p.terrainGridParams.w;
			push.terrainNumLevels = p.terrainGridParams.x;
			push.waterLevel = p.waterLevel;
			if (ServiceLocator::Instance().Has<IFoliageManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IFoliageManager>();
				enabled = mgr->IsEnabled();
				auto props = mgr->GetGlobalProperties();
				push.lodBaseRange = props.lodBaseRange;
				push.lodScaleFactor = props.lodScaleFactor;
				push.baseTileSize = props.baseTileSize;
				push.flowerRatio = props.flowerRatio;
				push.fernRatio = props.fernRatio;
				push.rockRatio = props.rockRatio;
				push.seaweedRatio = props.seaweedRatio;
				push.bushRatio = props.bushRatio;
				push.treeRatio = props.treeRatio;
				push.lengthMultiplier = props.lengthMultiplier;
				push.widthMultiplier = props.widthMultiplier;
				push.densityMultiplier = props.densityMultiplier;
				push.windMultiplier = props.windMultiplier;
				push.rigidityMultiplier = props.rigidityMultiplier;
				push.baseScale = props.baseScale;
				push.foliageFlags = (props.enableOcclusionCulling != 0u) ? 1u : 0u;

				push.gridParams.x = props.maxLODs;
				push.gridParams.y = props.tilesPerRow;
				push.gridParams.z = props.maxLODs * props.tilesPerRow * props.tilesPerRow;
			}
		}

		// No Modify<GBufferMaterial>() -- the attachment-count drift this fixes: the removed
		// Resources claimed the whole GBuffer<ModifyKey> group (all 5 keys) at compile time, but
		// this node's own realizations (and its pipeline's colorFormats below) only ever covered
		// 4 -- a real lesson #2 over-claim, now impossible to reintroduce by accident, since
		// there's no group alias forcing all 5 and every access here is hand-written to match
		// what Execute (and the pipeline request) actually does.
		//
		// ReadPrevious<HiZTexture>() replaces the old hand-rolled History<HiZTexture> realization
		// (HiZTextureDesc(ctx.width, ctx.height)) now that HiZTexture is catalog-converted (S1) --
		// the catalog is the real fix for lesson #9's "reader must get the desc right" trap, this
		// node was just the first one it was found on. Read<TerrainClipmapTexture/MinMaxTexture/
		// BiomeTexture/WeatherBiomeTexture>() are all safe now (the last one restored in S4 --
		// catalog-converted since batch 5, just never re-added then). Read<
		// FoliageBiomeTableTexture>() stays omitted -- not catalog-converted, and its producer
		// (a host-write node) schedules before this one, so an empty-desc realization here would
		// win via lesson #10 (see GTAONode.hpp's identical reasoning). AnyOrder() on all four GBuffer
		// keys here, matching EntityNode's own identical reasoning (see its Setup()'s comment) --
		// both nodes Modify these same four keys with nothing else ordering them, which the
		// relational compiler (S3) treats as a hard ambiguity error unless both sides opt in.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics).Active(enabled);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.ReadPrevious<HiZTexture>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<TerrainBiomeTexture>();
			r.Read<TerrainWeatherBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.hizIndex = ctx.Index<graph::History<HiZTexture>>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.biomeTableIndex = ctx.Index<FoliageBiomeTableTexture>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb
            };
			render::ResolvedPipeline resolved = render::BindGraphics(
				ctx,
				*pipelineLibrary,
				stages,
				kPipelineState,
				colorFormats,
				push,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				vk::Format::eD32Sfloat
			);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			uint32_t          taskGroupCount = (push.gridParams.z + 31) / 32;
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(FoliageNode);

} // namespace brassica
