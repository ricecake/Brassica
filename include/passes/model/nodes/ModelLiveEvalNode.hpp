#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "passes/model/nodes/ModelCullNode.hpp"
#include "passes/model/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	[[nodiscard]] inline ModelRegistry* ModelLiveEvalQueryRegistry() {
		if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
			return ServiceLocator::Instance().Get<ModelRegistry>().get();
		}
		return nullptr;
	}

	struct ModelLiveFinalizePushConstants {
		std::uint32_t liveIndirectArgsBufIdx{0};
	};

	static_assert(sizeof(ModelLiveFinalizePushConstants) == 4, "ModelLiveFinalizePushConstants size must be 4 bytes");

	// Compute-domain, single-thread, between ModelCullNode (writes rawLiveCount via atomicAdd)
	// and ModelLiveEvalNode (consumes groupCountX for its indirect dispatch) -- converts the raw
	// claim count into a real ceil(rawLiveCount/64) workgroup count. Split into its own node for
	// the same "let the graph insert the barrier between distinct nodes" reasoning as
	// ModelCullResetNode/ModelCullNode's own split (ModelCullNode.hpp) -- not a hand-rolled
	// intra-node barrier between two dispatches.
	struct ModelLiveFinalizeNode: render::NodeRegistrar<ModelLiveFinalizeNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/model_live_finalize.comp")) {
				spdlog::critical("ModelLiveFinalizeNode shader compilation failed.");
				throw std::runtime_error("ModelLiveFinalizeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = ModelLiveEvalQueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(totalMeshletCount > 0);
			r.Modify<ModelLiveIndirectArgsBuffer>();
			// Explicit, not left to Modify-vs-Modify tie-breaking: this really is a required
			// order (read rawLiveCount after ModelCullNode's atomicAdd writes it), not a
			// commutative AnyOrder() case.
			r.After<ModelCullNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			ModelLiveFinalizePushConstants push{};
			push.liveIndirectArgsBufIdx = ctx.BufferIndex<ModelLiveIndirectArgsBuffer>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(1, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ModelLiveFinalizeNode);

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6).
	struct ModelLiveEvalPushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t modelDescBufIdx{0};
		std::uint32_t skeletonBufIdx{0};
		std::uint32_t liveListBufIdx{0};
		std::uint32_t liveIndirectArgsBufIdx{0};
		std::uint32_t livePaletteBufIdx{0};
	};

	static_assert(sizeof(ModelLiveEvalPushConstants) == 24, "ModelLiveEvalPushConstants size must be 24 bytes");

	// Stage 4: evaluates every instance ModelCullNode claimed a live-pool slot for this frame --
	// one thread per live instance (model_live_eval.comp), writing global*inverseBind into
	// ModelLivePaletteBuffer. Indirect dispatch, sized by ModelLiveFinalizeNode's own output
	// (ceil(rawLiveCount/64) workgroups) -- Active alongside ModelCullNode/ModelRenderNode
	// (same totalMeshletCount gate), not gated on "is there a live-capable instance this frame":
	// that distinction is handled entirely by the indirect dispatch's own groupCountX (0 on a
	// frame with no live claims), the same way ModelRenderNode's dispatch already degrades to
	// nothing when ModelCullNode's visible count is 0.
	struct ModelLiveEvalNode: render::NodeRegistrar<ModelLiveEvalNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/model_live_eval.comp")) {
				spdlog::critical("ModelLiveEvalNode shader compilation failed.");
				throw std::runtime_error("ModelLiveEvalNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<ModelLivePaletteBuffer>();
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         reg = ModelLiveEvalQueryRegistry();
			std::uint32_t totalMeshletCount = reg ? reg->GetTotalMeshletCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(totalMeshletCount > 0);
			r.Read<ModelInstanceBuffer>();
			r.Read<ModelDescBuffer>();
			r.Read<ModelSkeletonBuffer>();
			r.Read<ModelLiveListBuffer>();
			r.Read<ModelLiveIndirectArgsBuffer>();
			r.Create<ModelLivePaletteBuffer>();
			r.After<ModelLiveFinalizeNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}

			// Same "not provisioned until first-ever spawn" guard as ModelCullNode/
			// ModelRenderNode's own (see ModelRenderNode::Execute's comment).
			if (!physReg->GetBuffer<ModelInstanceBuffer>()) {
				return;
			}

			ModelLiveEvalPushConstants push{};
			push.instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			push.modelDescBufIdx = ctx.BufferIndex<ModelDescBuffer>();
			push.skeletonBufIdx = ctx.BufferIndex<ModelSkeletonBuffer>();
			push.liveListBufIdx = ctx.BufferIndex<ModelLiveListBuffer>();
			push.liveIndirectArgsBufIdx = ctx.BufferIndex<ModelLiveIndirectArgsBuffer>();
			push.livePaletteBufIdx = ctx.BufferIndex<ModelLivePaletteBuffer>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::Buffer    indirectBuf{nullptr};
			std::uint64_t indirectOffset = 0;
			if (auto physBuf = physReg->GetBuffer<ModelLiveIndirectArgsBuffer>()) {
				indirectBuf = physBuf->GetBuffer();
				indirectOffset = physBuf->SliceStride() * (ctx.frameIndex % physBuf->RingSlots());
			}
			if (!indirectBuf) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatchIndirect(indirectBuf, indirectOffset);
		}
	};

	BRASSICA_REGISTER_NODE(ModelLiveEvalNode);

} // namespace brassica
