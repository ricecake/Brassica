#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/ModelTypes.hpp"
#include "passes/model/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Compute-domain, sole Create<>() of ModelIndirectArgsBuffer -- split from ModelCullNode
	// below for the same reason FoliageIndirectResetNode is split from FoliageNode (see that
	// node's own comment): this reset's FillBufferRegions issues a real vkCmdFillBuffer, which
	// the generic graph machinery only knows how to barrier correctly against a *different*
	// node's later access to the same key, not a second manual barrier hand-rolled within one
	// node's own Execute(). Active alongside ModelCullNode/ModelRenderNode (same
	// totalMeshletCount gate), not alongside ModelInstanceManager's own spawn activity -- this
	// must reset every active frame regardless of whether anything spawned this frame, since
	// model_cull.comp's atomicAdd needs a freshly-zeroed counter every time it runs.
	struct ModelCullResetNode: render::NodeRegistrar<ModelCullResetNode> {
		void Init(const render::NodeServices& services) {
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<ModelIndirectArgsBuffer>();
				services.physicalRegistry->Catalog().Get<ModelLiveIndirectArgsBuffer>();
			}
		}

		void Destroy(vk::Device) {}

		[[nodiscard]] static ModelRegistry* QueryRegistry() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
				return ServiceLocator::Instance().Get<ModelRegistry>().get();
			}
			return nullptr;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = QueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(totalMeshletCount > 0);
			r.Create<ModelIndirectArgsBuffer>();
			r.Create<ModelLiveIndirectArgsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			if (auto buf = physReg->GetBuffer<ModelIndirectArgsBuffer>()) {
				// Same layout/reasoning as FoliageIndirectResetNode's own reset: groupCountX
				// is model_cull.comp's live survivor counter (zeroed here), Y/Z are the
				// fixed-at-1 dispatch dimensions this draw never varies.
				const std::array<graph::BufferFillRegion, 2> regions{
					graph::BufferFillRegion{0, 4, 0u},
					graph::BufferFillRegion{4, 8, 1u},
				};
				graph::FillBufferRegions(vkCmd, *buf, graph::AccessKind::ReadWrite, ctx.domain, regions);
			}

			if (auto buf = physReg->GetBuffer<ModelLiveIndirectArgsBuffer>()) {
				// groupCountX/Y/Z get overwritten by ModelLiveFinalizeNode once rawLiveCount is
				// known (word 12) -- zeroing all four here is simplest and safe either way.
				const std::array<graph::BufferFillRegion, 1> regions{
					graph::BufferFillRegion{0, 16, 0u},
				};
				graph::FillBufferRegions(vkCmd, *buf, graph::AccessKind::ReadWrite, ctx.domain, regions);
			}
		}
	};

	BRASSICA_REGISTER_NODE(ModelCullResetNode);

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6).
	struct ModelCullPushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t modelDescBufIdx{0};
		std::uint32_t clipTableBufIdx{0};
		std::uint32_t visibleListBufIdx{0};
		std::uint32_t indirectArgsBufIdx{0};
		std::uint32_t liveListBufIdx{0};         // Stage 4
		std::uint32_t liveIndirectArgsBufIdx{0}; // Stage 4
	};

	static_assert(sizeof(ModelCullPushConstants) == 28, "ModelCullPushConstants size must be 28 bytes");

	// Stage 3/4: walks every ModelInstanceBuffer slot (same fixed kModelInstanceCapacity walk
	// model_render.task used to do directly), resolving pose and frustum-culling each live
	// instance, and (Stage 4) picking a render mode per survivor -- see model_cull.comp's own
	// comment for the full capability/distance/pool-budget selection -- then compacting
	// survivors into ModelVisibleListBuffer instead of a per-task-workgroup local array.
	// ModelRenderNode's task shader becomes a thin, indirectly-dispatched consumer of this
	// node's output (one task workgroup per compacted survivor,
	// matching EntityNode's own indirect task+mesh dispatch shape).
	struct ModelCullNode: render::NodeRegistrar<ModelCullNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/model_cull.comp")) {
				spdlog::critical("ModelCullNode shader compilation failed.");
				throw std::runtime_error("ModelCullNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<ModelVisibleListBuffer>();
				services.physicalRegistry->Catalog().Get<ModelLiveListBuffer>();
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		[[nodiscard]] static ModelRegistry* QueryRegistry() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
				return ServiceLocator::Instance().Get<ModelRegistry>().get();
			}
			return nullptr;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = QueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(totalMeshletCount > 0);
			r.Read<ModelInstanceBuffer>();
			r.Read<ModelDescBuffer>();
			r.Read<ModelClipTableBuffer>();
			r.Create<ModelVisibleListBuffer>();
			r.Modify<ModelIndirectArgsBuffer>();
			r.Create<ModelLiveListBuffer>();
			r.Modify<ModelLiveIndirectArgsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}

			// ModelInstanceBuffer is only provisioned once the first spawn ever happens -- same
			// guard, same reasoning, as ModelRenderNode::Execute's own (see that node's comment).
			// Before that, there's nothing to cull: ModelIndirectArgsBuffer stays at the
			// all-zero state ModelCullResetNode just wrote, so ModelRenderNode's indirect dispatch
			// correctly draws nothing either.
			if (!physReg->GetBuffer<ModelInstanceBuffer>()) {
				return;
			}

			ModelCullPushConstants push{};
			push.instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			push.modelDescBufIdx = ctx.BufferIndex<ModelDescBuffer>();
			push.clipTableBufIdx = ctx.BufferIndex<ModelClipTableBuffer>();
			push.visibleListBufIdx = ctx.BufferIndex<ModelVisibleListBuffer>();
			push.indirectArgsBufIdx = ctx.BufferIndex<ModelIndirectArgsBuffer>();
			push.liveListBufIdx = ctx.BufferIndex<ModelLiveListBuffer>();
			push.liveIndirectArgsBufIdx = ctx.BufferIndex<ModelLiveIndirectArgsBuffer>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t     groupCount = (kModelInstanceCapacity + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ModelCullNode);

} // namespace brassica
