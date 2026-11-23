#pragma once

#include "graph/Execution.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/model/DecorAccelerationStructure.hpp"
#include "passes/model/ModelRegistry.hpp"
#include "render/NodeLifecycle.hpp"
#include "ServiceLocator.hpp"

namespace brassica {

	// Drives DecorAccelerationStructure::RebuildIfNeeded every frame -- runs as a real node
	// (rather than an Engine::Init()-time one-shot, the way the deleted
	// TerrainAccelerationStructure's own InitAccelerationStructures was) purely to get a valid
	// vk::CommandBuffer to record BLAS builds into at the right point in the frame.
	// RebuildIfNeeded itself no-ops on every frame where ModelRegistry's own version hasn't
	// advanced (the common case); it reads registry.GetVertices()/GetFlatTriangleIndices()
	// directly off the CPU-side registry, not any graph-managed GPU resource, so this node needs
	// no resource edges at all.
	struct DecorTemplateBlasNode: render::NodeRegistrar<DecorTemplateBlasNode> {
		void Init(const render::NodeServices&) {}

		// DecorAccelerationStructure is Engine-owned (destroyed in Engine::Cleanup) -- this node
		// holds no Vulkan resources of its own.
		void Destroy(vk::Device) {}

		[[nodiscard]] static ModelRegistry* QueryRegistry() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<ModelRegistry>()) {
				return ServiceLocator::Instance().Get<ModelRegistry>().get();
			}
			return nullptr;
		}

		[[nodiscard]] static DecorAccelerationStructure* QueryDecorAS() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<DecorAccelerationStructure>()) {
				return ServiceLocator::Instance().Get<DecorAccelerationStructure>().get();
			}
			return nullptr;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			auto* reg = QueryRegistry();
			r.Domain(graph::ExecutionDomain::Compute).Active(reg && reg->ModelCount() > 0);
		}

		void Execute(graph::NodeContext& ctx) {
			auto* reg = QueryRegistry();
			auto* decorAS = QueryDecorAS();
			if (!reg || !decorAS) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			decorAS->RebuildIfNeeded(vkCmd, *reg);
		}
	};

	BRASSICA_REGISTER_NODE(DecorTemplateBlasNode);

} // namespace brassica
