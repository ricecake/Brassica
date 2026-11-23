#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/ModelInstanceManager.hpp"
#include "passes/model/resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6 -- see
	// shaders/model_instance.glsl's own comment), not device addresses.
	struct ModelInstanceScatterPushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t writeBufIdx{0};
		std::uint32_t writeCount{0};
	};

	static_assert(sizeof(ModelInstanceScatterPushConstants) == 12, "ModelInstanceScatterPushConstants size must be 12 bytes");

	// Compute-domain sole Create<>() of ModelInstanceBuffer -- applies ModelInstanceManager's
	// pending sparse writes (see that class's own comment) via a GPU scatter, the CPU-driven
	// path onto the table (Stage 5.2's decor population writes directly from the GPU instead,
	// bypassing this node's scatter dispatch but still relying on its one-time zero-clear).
	// Unconditionally active -- see Setup()'s own comment for why this isn't gated on pending
	// writes the way it used to be.
	struct ModelInstanceScatterNode: render::NodeRegistrar<ModelInstanceScatterNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		bool                     initialized{false};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/model_instance_scatter.comp")) {
				spdlog::critical("ModelInstanceScatterNode shader compilation failed.");
				throw std::runtime_error("ModelInstanceScatterNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<ModelInstanceBuffer>();
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		[[nodiscard]] static ModelInstanceManager* QueryManager() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelInstanceManager>()) {
				return ServiceLocator::Instance().Get<ModelInstanceManager>().get();
			}
			return nullptr;
		}

		// Unconditionally active, not gated on "the CPU has pending writes" (the old gate) or on
		// any other service's state -- Stage 5.2's decor population writes directly into
		// ModelInstanceBuffer from the GPU, with no CPU spawn call ever involved, so this node's
		// Create<>() (and the one-time zero-clear in Execute() below) must not depend on a
		// manual spawn ever having happened. Tried gating on ModelRegistry::GetTotalMeshletCount
		// first (matching ModelRenderNode/ModelCullNode's own gate) but that coupled this node to
		// a service that isolated tests (e.g. tests/test_model_instance_scatter_node.cpp) don't
		// register -- a real regression caught by that existing test, not hypothetical.
		// Unconditional is simpler and has no real cost: Execute() below still exits cheaply on
		// a quiet frame with nothing to scatter.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(true);
			r.Create<ModelInstanceBuffer>();
			r.Read<ModelInstanceWriteBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}

			auto instanceBuf = physReg->GetBuffer<ModelInstanceBuffer>();
			if (!instanceBuf) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// First time this buffer is ever provisioned, its contents are undefined -- every
			// slot nobody writes (CPU scatter below, or Stage 5.2's GPU-direct decor writes)
			// must still read as all-zero (dead, rest pose) rather than garbage. Only needed
			// once: kLifetime::Persistent plus desc.persistent keep it alive and correctly
			// flagged defined from here on (see ModelInstanceBuffer's own comment). Decoupled
			// from the writeCount-gated scatter dispatch below on purpose -- see this struct's
			// own Setup() comment on why.
			if (!initialized) {
				graph::ClearBufferToZero(vkCmd, *instanceBuf, graph::AccessKind::ReadWrite, ctx.domain);
				initialized = true;
			}

			auto* mgr = QueryManager();
			std::uint32_t writeCount = mgr ? mgr->GetLastFlushCount() : 0u;
			if (writeCount == 0) {
				return;
			}

			if (!physReg->GetBuffer<ModelInstanceWriteBuffer>()) {
				return;
			}

			ModelInstanceScatterPushConstants push{};
			push.writeCount = writeCount;
			push.instanceBufIdx = ctx.BufferIndex<ModelInstanceBuffer>();
			push.writeBufIdx = ctx.BufferIndex<ModelInstanceWriteBuffer>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			std::uint32_t groupCount = (writeCount + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ModelInstanceScatterNode);

} // namespace brassica
