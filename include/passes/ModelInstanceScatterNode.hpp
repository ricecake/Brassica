#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalExecutionBackend.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "model/ModelInstanceManager.hpp"
#include "passes/ResourceKeys.hpp"
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
	// pending sparse writes (see that class's own comment) via a GPU scatter, the only path
	// that ever touches the table. Active only on frames with at least one pending write; a
	// quiet frame contributes nothing to the graph at all, same shape as
	// FoliageIndirectResetNode/ParticleResetNode's own activity-gated Setup().
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

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto*         mgr = QueryManager();
			std::uint32_t writeCount = mgr ? mgr->GetLastFlushCount() : 0u;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(writeCount > 0);
			r.Create<ModelInstanceBuffer>();
			r.Read<ModelInstanceWriteBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			auto* mgr = QueryManager();
			if (!mgr) {
				return;
			}
			std::uint32_t writeCount = mgr->GetLastFlushCount();
			if (writeCount == 0) {
				return;
			}

			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			auto instanceBuf = physReg->GetBuffer<ModelInstanceBuffer>();
			if (!instanceBuf) {
				return;
			}

			// First time this buffer is ever provisioned, its contents are undefined -- every
			// slot this frame's writes don't touch must still read as all-zero (dead, rest
			// pose) rather than garbage. Only needed once: kLifetime::Persistent plus
			// desc.persistent keep it alive and correctly flagged defined from here on (see
			// ModelInstanceBuffer's own comment).
			if (!initialized) {
				graph::ClearBufferToZero(vkCmd, *instanceBuf, graph::AccessKind::ReadWrite, ctx.domain);
				initialized = true;
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
