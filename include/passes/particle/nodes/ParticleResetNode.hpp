#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/particle/IParticleManager.hpp"
#include "passes/particle/ParticleCommon.hpp"
#include "passes/RenderPhases.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "resources.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct ParticleResetPushConstants {
		std::uint32_t gridSize{1024};
	};

	// S4: no kPhase -- this node is the sole Create<> of all three of its buffers, so
	// Create-before-accessor already orders it before every other toucher (ParticleLivenessNode/
	// ParticleGridBuildNode/ParticleBehaviorNode) regardless of phase.
	//
	// S6: promoted out of ParticleSystemNode's Subgraph to be an independent top-level node, same
	// as UnderwaterParticleRenderNode/AboveWaterParticleRenderNode already were -- the relation-
	// based scheduler doesn't need the inner-Graph indirection a Subgraph gave the old
	// phase-partitioned compiler, so this node now owns its own descriptor set (standard
	// Init(services) instead of a custom multi-arg signature; the old "one shared cache" dance
	// existed only to avoid two nodes double-updating one *shared* vk::DescriptorSet -- with each
	// node owning its own set outright, same as the two render nodes, that coordination problem
	// doesn't exist to solve).
	struct ParticleResetNode: render::NodeRegistrar<ParticleResetNode> {
		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		std::uint32_t                   gridSize{1024};
		detail::ParticleDescriptorSet   descriptorSet{};
		detail::ParticleDescriptorCache descriptorCache{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			descriptorSet = detail::CreateParticleDescriptorSet(services.device);
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_reset.comp")) {
				spdlog::critical("ParticleResetNode shader compilation failed.");
				throw std::runtime_error("ParticleResetNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
			detail::DestroyParticleDescriptorSet(device, descriptorSet);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<graph::PhysicalResourceRegistry>()) {
				auto registry = ServiceLocator::Instance().Get<graph::PhysicalResourceRegistry>();
				registry->Catalog().Get<ParticleGridHeadsBuffer>().byteSize = gridSize * sizeof(int);
			}

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			// AboveWater/UnderwaterParticleIndirectBuffer/ParticleGridHeadsBuffer are all
			// catalog-converted -- no .Desc() needed, see passes/particle/resources.hpp.
			r.Create<AboveWaterParticleIndirectBuffer>();
			r.Create<UnderwaterParticleIndirectBuffer>();
			r.Create<ParticleGridHeadsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(
						descriptorCache,
						registry->GetDevice(),
						descriptorSet.set,
						registry
					);
				}
			}

			ParticleResetPushConstants             push{.gridSize = gridSize};
			std::array<vk::DescriptorSetLayout, 1> extraLayouts{descriptorSet.layout};
			std::array<vk::DescriptorSet, 1>       extraSets{descriptorSet.set};
			render::ResolvedPipeline               resolved = render::BindCompute(
                ctx,
                *pipelineLibrary,
                compShader,
                push,
                render::ExtraSets{extraLayouts, extraSets}
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t     groupCount = (gridSize + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ParticleResetNode);

} // namespace brassica
