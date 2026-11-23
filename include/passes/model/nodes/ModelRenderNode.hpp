#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>
#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/material/resources.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/resources.hpp"
#include "passes/scene/resources.hpp"
#include "render/NodeCommands.hpp"
#include "resources/core.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6 -- see
	// shaders/model_instance.glsl's own comment), not device addresses: every buffer here is
	// eStorageBuffer-usage and already auto-registered by
	// PhysicalRegistry::AssignAndWriteBindlessBufferIndex, so there's no address to fetch at all.
	// No per-model jointCount/meshletCount here anymore (Stage 1 had them, single-model only) --
	// every model's own values now live in ModelDescBuffer, looked up per-survivor by modelIndex
	// (model_render.task/mesh).
	struct ModelRenderPushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t modelDescBufIdx{0};
		std::uint32_t clipTableBufIdx{0};
		std::uint32_t vertexBufIdx{0};
		std::uint32_t meshletVertexBufIdx{0};
		std::uint32_t meshletTriangleBufIdx{0};
		std::uint32_t meshletBufIdx{0};
		std::uint32_t paletteBufIdx{0};
		std::uint32_t materialBufferIndex{0};
		std::uint32_t visibleListBufIdx{0}; // Stage 3: ModelCullNode's compacted survivor list
		std::uint32_t livePaletteBufIdx{0}; // Stage 4: ModelLiveEvalNode's transient live palette
	};

	static_assert(sizeof(ModelRenderPushConstants) == 44, "ModelRenderPushConstants size must be 44 bytes");

	// Matches shaders/model_instance.glsl's ModelIndirectArgsBufferArray (and
	// VkDrawMeshTasksIndirectCommandEXT) exactly -- only needed here for its sizeof(), as the
	// indirect draw's stride (same role as FoliageIndirectArgsLayout/MeshTasksIndirectCommand).
	struct ModelIndirectArgsLayout {
		std::uint32_t groupCountX;
		std::uint32_t groupCountY;
		std::uint32_t groupCountZ;
	};

	static_assert(sizeof(ModelIndirectArgsLayout) == 12, "ModelIndirectArgsLayout size must be 12 bytes");

	// Successor to GltfRenderNode: draws every live slot in ModelInstanceBuffer instead of one
	// hardcoded instance. Stage 3 moved the per-slot walk, pose resolution, and frustum culling
	// out of model_render.task entirely, into ModelCullNode (passes/ModelCullNode.hpp) -- this
	// node's task shader is now a thin consumer, dispatched indirectly over exactly
	// ModelCullNode's compacted survivor count (ModelIndirectArgsBuffer), one task workgroup per
	// survivor. meshletCount/jointCount come from ModelRegistry via the same ServiceLocator
	// indirection GltfRenderNode used for GltfModelManager.
	struct ModelRenderNode: render::NodeRegistrar<ModelRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = true,
		};

		render::PipelineLibrary*               pipelineLibrary = nullptr;
		TaskShader                              taskShader;
		MeshShader                              meshShader;
		FragmentShader                          fragShader;
		const brassica::DispatchLoaderDynamic* dls = nullptr;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/model_render.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/model_render.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/gltf.frag")) {
				spdlog::critical("ModelRenderNode shader compilation failed.");
				throw std::runtime_error("ModelRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&taskShader);
				services.shaderWatcher->RegisterShader(&meshShader);
				services.shaderWatcher->RegisterShader(&fragShader);
			}
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		[[nodiscard]] static ModelRegistry* QueryRegistry() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
				return ServiceLocator::Instance().Get<ModelRegistry>().get();
			}
			return nullptr;
		}

		// Bare Read<>() on the four GLTF geometry buffers, the two clip buffers, and
		// ModelDescBuffer: each has exactly one real producer (ModelRegistry's own
		// PredefinedBufferNode members), same non-catalog reasoning as GltfRenderNode's own
		// Setup() had. ModelInstanceBuffer is catalog-converted (ResourceKeys.hpp) and
		// kLifetime::Persistent -- its bare Read<>() here gets a real desc from the catalog
		// regardless, and is exempt from zero-init-reads on a frame with no pending
		// spawns/clip changes (see that key's own comment).
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = QueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Graphics).Active(totalMeshletCount > 0);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferMaterial>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.Read<GltfVertexBuffer>();
			r.Read<GltfMeshletVertexBuffer>();
			r.Read<GltfMeshletTriangleBuffer>();
			r.Read<GltfMeshletBuffer>();
			r.Read<ModelClipPaletteBuffer>();
			r.Read<ModelClipTableBuffer>();
			r.Read<ModelDescBuffer>();
			r.Read<ModelInstanceBuffer>();
			r.Read<MaterialBuffer>();
			r.Read<ModelVisibleListBuffer>();
			r.Read<ModelIndirectArgsBuffer>();
			r.Read<ModelLivePaletteBuffer>();
			r.ReadPrevious<ShadingRateMap>().ShadingRateTexelSize({s_shadingRateTileWidth, s_shadingRateTileHeight});
		}

		void Execute(graph::NodeContext& ctx) {
			auto* reg = QueryRegistry();
			if (!reg || reg->GetTotalMeshletCount() == 0) {
				return;
			}

			// ModelInstanceBuffer is only provisioned once the first spawn ever happens
			// (ModelInstanceScatterNode is its sole Create<>(), active only on frames with
			// pending writes) -- before that, there's nothing to draw yet. Checked directly via
			// GetBuffer<>(), not via BufferIndex<>() == 0: a bindless index of 0 is a perfectly
			// valid real slot (unlike a device address, 0 isn't a sentinel here), so the only
			// reliable "not provisioned yet" signal is the pointer itself.
			if (auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (!physReg->GetBuffer<ModelInstanceBuffer>()) {
					return;
				}
			}

			ModelRenderPushConstants push{};
			push.materialBufferIndex = ctx.BufferIndex<MaterialBuffer>();
			push.instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			push.modelDescBufIdx = ctx.BufferIndex<ModelDescBuffer>();
			push.clipTableBufIdx = ctx.BufferIndex<ModelClipTableBuffer>();
			push.vertexBufIdx = ctx.BufferIndex<GltfVertexBuffer>();
			push.meshletVertexBufIdx = ctx.BufferIndex<GltfMeshletVertexBuffer>();
			push.meshletTriangleBufIdx = ctx.BufferIndex<GltfMeshletTriangleBuffer>();
			push.meshletBufIdx = ctx.BufferIndex<GltfMeshletBuffer>();
			push.paletteBufIdx = ctx.BufferIndex<ModelClipPaletteBuffer>();
			push.visibleListBufIdx = ctx.BufferIndex<ModelVisibleListBuffer>();
			push.livePaletteBufIdx = ctx.BufferIndex<ModelLivePaletteBuffer>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb,
                vk::Format::eR8G8B8A8Unorm
            };
			render::ResolvedPipeline resolved = render::BindGraphics(
				ctx,
				*pipelineLibrary,
				stages,
				kPipelineState,
				colorFormats,
				push,
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				vk::Format::eD32SfloatS8Uint
			);
			if (!resolved.pipeline) {
				return;
			}

			vk::Buffer    indirectBuf{nullptr};
			std::uint64_t indirectOffset = 0;
			if (auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (auto physBuf = physReg->GetBuffer<ModelIndirectArgsBuffer>()) {
					indirectBuf = physBuf->GetBuffer();
					indirectOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
				}
			}

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					indirectOffset,
					1,
					sizeof(ModelIndirectArgsLayout)
				);
			}
		}
	};

	BRASSICA_REGISTER_NODE(ModelRenderNode);

} // namespace brassica
