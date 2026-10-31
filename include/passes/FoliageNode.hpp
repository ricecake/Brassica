#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

#include "constants.h"
#include "foliage/IFoliageManager.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
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

	// kFoliageInstanceCapacity lives in ResourceKeys.hpp now, alongside FoliageInstanceBuffer's
	// catalog-converted Desc() that needs it too.

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
		// Appended, not interleaved -- see feedback_push_constant_struct_editing. Device
		// addresses for FoliageInstanceBuffer/FoliageIndirectArgsBuffer (shaders/
		// foliage_instance.glsl); foliage.task's own copy of this struct doesn't need these and
		// isn't updated to match, since it never touches either buffer.
		alignas(8) std::uint64_t instanceBufAddress{0};
		alignas(8) std::uint64_t indirectArgsBufAddress{0};
	};

	static_assert(sizeof(FoliagePushConstants) == 136, "FoliagePushConstants size must be 136 bytes");

	// Compute-domain, sole Create<>() of FoliageIndirectArgsBuffer -- FoliageNode below now
	// Modify<>()s it instead of creating it, same split as ParticleResetNode/ParticleLivenessNode
	// (ParticleSystemNode.hpp). Split out specifically because this buffer's live append counter
	// (groupCountX, which foliage.mesh atomicAdd's into every frame) needs zeroing before that
	// collection runs, and graph::ClearBufferToZero issues a real vkCmdPipelineBarrier2 + a real
	// vkCmdFillBuffer -- both illegal inside a dynamic-rendering instance
	// (VUID-vkCmdPipelineBarrier2-None-09553/09554, VUID-vkCmdFillBuffer-renderpass). FoliageNode
	// itself is Graphics-domain, and PhysicalExecutionBackend::Execute calls
	// DynamicRenderingWrapper::Begin for every Graphics-domain node *before* that node's own
	// Execute() runs (unconditionally, based on its own GBuffer realizations) -- so calling
	// ClearBufferToZero from inside FoliageNode::Execute(), as this used to do, was always
	// illegal. Confirmed via real GPU-assisted validation on real hardware, not hypothetical.
	// Compute-domain nodes never get a dynamic-rendering scope opened around them at all
	// (DynamicRenderingWrapper::Begin early-returns for any non-Graphics domain), which is what
	// actually fixes this, not just moving the call around within the same node.
	struct FoliageIndirectResetNode: render::NodeRegistrar<FoliageIndirectResetNode> {
		// Registers FoliageIndirectArgsBuffer's catalog Desc() here, not (only) in FoliageNode's
		// own Init() below -- this node is the key's real sole Create<>() now, so it should be
		// the one responsible for the catalog touch, not a node that merely Modify<>()s it
		// afterward. Found as a real gap, not hypothetical, writing a standalone test for this
		// node in isolation (no FoliageNode in the same graph) that reproduced the exact
		// "resolved to neither a texture, a buffer, nor an acceleration structure" crash this
		// catalog touch exists to prevent -- the whole reason that crash ever went away in the
		// real engine was FoliageNode's Init() happening to also run and incidentally registering
		// the same key, not this node correctly owning its own registration.
		void Init(const render::NodeServices& services) {
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<FoliageIndirectArgsBuffer>();
			}
		}

		void Destroy(vk::Device) {}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IFoliageManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IFoliageManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}
			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			r.Create<FoliageIndirectArgsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (auto buf = physReg->GetBuffer<FoliageIndirectArgsBuffer>()) {
					vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
					graph::ClearBufferToZero(vkCmd, *buf, graph::AccessKind::ReadWrite, ctx.domain);

					// Root cause of archetype-rendered trees/bushes never appearing: this buffer
					// is a VkDrawMeshTasksIndirectCommandEXT{groupCountX,Y,Z}, and
					// ClearBufferToZero above zeroes all of it -- but foliage.mesh's own
					// collection only ever atomicAdd's groupCountX (shaders/foliage.mesh);
					// nothing anywhere writes groupCountY/Z. A Vulkan indirect dispatch's real
					// total is groupCountX*Y*Z, so Y=Z=0 means FoliageArchetypeRenderNode's
					// drawMeshTasksIndirectEXT always dispatches zero workgroups, regardless of
					// how many instances foliage.mesh actually collected. This predates this
					// session's own work (the buffer layout/zeroing was always this shape) but
					// was masked for as long as foliage.mesh's now-removed emitTreeItem/
					// emitBushItem calls drew trees/bushes live, independent of this buffer --
					// once that fallback was cut over to archetype-only rendering, the
					// always-broken indirect path became the only path, and trees/bushes
					// stopped appearing at all. Patched here with a small vkCmdUpdateBuffer
					// instead of changing ClearBufferToZero itself, which is a shared utility
					// other callers rely on zeroing *everything*.
					const auto& desc = buf->GetDesc();
					if (desc.usageMask & static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eTransferDst)) {
						vk::BufferMemoryBarrier2 toTransferDst{
							buf->GetLastStage(),
							buf->GetLastAccess(),
							vk::PipelineStageFlagBits2::eAllTransfer,
							vk::AccessFlagBits2::eTransferWrite,
							VK_QUEUE_FAMILY_IGNORED,
							VK_QUEUE_FAMILY_IGNORED,
							buf->GetBuffer(),
							4,
							8,
						};
						vk::DependencyInfo preDep{};
						preDep.setBufferMemoryBarriers(toTransferDst);
						vkCmd.pipelineBarrier2(preDep);

						const std::array<std::uint32_t, 2> groupCountYZ{1u, 1u};
						vkCmd.updateBuffer<std::uint32_t>(buf->GetBuffer(), 4, groupCountYZ);

						graph::ResourceState postUpdate = graph::DeriveBufferState(
							graph::AccessKind::ReadWrite,
							ctx.domain,
							desc.usageMask ? vk::BufferUsageFlags(desc.usageMask) : vk::BufferUsageFlags{}
						);
						vk::BufferMemoryBarrier2 toFinal{
							vk::PipelineStageFlagBits2::eAllTransfer,
							vk::AccessFlagBits2::eTransferWrite,
							postUpdate.stage,
							postUpdate.access,
							VK_QUEUE_FAMILY_IGNORED,
							VK_QUEUE_FAMILY_IGNORED,
							buf->GetBuffer(),
							4,
							8,
						};
						vk::DependencyInfo postDep{};
						postDep.setBufferMemoryBarriers(toFinal);
						vkCmd.pipelineBarrier2(postDep);
						buf->SetLastStageAccess(postUpdate.stage, postUpdate.access);
					}
				}
			}
		}
	};

	BRASSICA_REGISTER_NODE(FoliageIndirectResetNode);

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

			// Same registration gap, same fix, as FoliageBakeNode's own comment for
			// FoliageArchetypeVertexBuffer et al. -- FoliageInstanceBuffer is this node's own
			// catalog-converted, sole-Create<>()'d key (ResourceKeys.hpp) and needs the same
			// one-time catalog.Get<K>() touch or its Desc() is never consulted.
			// FoliageIndirectArgsBuffer's own touch moved to FoliageIndirectResetNode::Init()
			// above -- that node is its real sole Create<>() now, not this one (this node only
			// Modify<>()s it); see that node's own comment for why registering it here instead
			// was a real, found gap, not just a style preference.
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<FoliageInstanceBuffer>();
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
		// BiomeTexture/WeatherBiomeTexture/FoliageBiomeTableTexture>() are all safe despite none
		// of the last one being catalog-converted: PhysicalRegistry.hpp's Provision() ignores an
		// empty-desc realization for provisioning purposes regardless of schedule order (an
		// un-Desc()'d Read<>() carries one by construction), so this reader can never stomp the
		// real producer's own desc no matter which one schedules later -- the older "empty-desc
		// wins via lesson #10" hazard that justified omitting FoliageBiomeTableTexture here no
		// longer applies once that guard exists. AnyOrder() on all four GBuffer keys here,
		// matching EntityNode's own identical reasoning (see its Setup()'s comment) -- both nodes
		// Modify these same four keys with nothing else ordering them, which the relational
		// compiler (S3) treats as a hard ambiguity error unless both sides opt in.
		// Per-frame instance collection (see shaders/foliage_instance.glsl). FoliageInstanceBuffer
		// is catalog-converted (ResourceKeys.hpp) -- bare Create<>(), no .Desc() here.
		// FoliageIndirectArgsBuffer is now Modify<>(), not Create<>() -- FoliageIndirectResetNode
		// above is its real sole Create<>() (Compute-domain, zeroes the live counter before this
		// node's collection runs); see its own comment for why that split exists at all.
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
			r.Read<FoliageBiomeTableTexture>();
			r.Create<FoliageInstanceBuffer>();
			r.Modify<FoliageIndirectArgsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.hizIndex = ctx.Index<graph::History<HiZTexture>>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.biomeTableIndex = ctx.Index<FoliageBiomeTableTexture>();

			// The live counter (groupCountX) itself is zeroed by FoliageIndirectResetNode now,
			// before this node runs (Create-before-accessor on FoliageIndirectArgsBuffer) -- see
			// its own comment for why that can't happen here anymore (this node is Graphics-domain
			// and already has its own dynamic-rendering scope open by the time Execute() runs).
			if (auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (auto buf = physReg->GetBuffer<FoliageInstanceBuffer>()) {
					vk::BufferDeviceAddressInfo info(buf->GetBuffer());
					push.instanceBufAddress = physReg->GetDevice().getBufferAddress(info);
				}
				if (auto buf = physReg->GetBuffer<FoliageIndirectArgsBuffer>()) {
					vk::BufferDeviceAddressInfo info(buf->GetBuffer());
					push.indirectArgsBufAddress = physReg->GetDevice().getBufferAddress(info);
				}
			}

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
