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

	struct ParticleGridBuildPushConstants {
		std::uint32_t maxParticles{8192};
		std::uint32_t gridSize{1024};
		float         cellSize{8.0f};
	};

	// S4: no kPhase -- ParticleResetNode's sole Create<> of ParticleGridHeadsBuffer orders it
	// before this node's Modify<> via Create-before-accessor; this node is in turn the sole
	// Create<> of ParticleGridNextBuffer, ordering it before ParticleBehaviorNode's Read<> of
	// both grid buffers.
	struct ParticleGridBuildNode: render::NodeRegistrar<ParticleGridBuildNode> {
		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		std::uint32_t                   maxParticles{8192};
		std::uint32_t                   gridSize{1024};
		float                           cellSize{8.0f};
		detail::ParticleDescriptorSet   descriptorSet{};
		detail::ParticleDescriptorCache descriptorCache{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			descriptorSet = detail::CreateParticleDescriptorSet(services.device);
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_grid_build.comp")) {
				spdlog::critical("ParticleGridBuildNode shader compilation failed.");
				throw std::runtime_error("ParticleGridBuildNode shader compilation failed.");
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
				auto& catalog = registry->Catalog();
				catalog.Get<ParticleGridHeadsBuffer>().byteSize = gridSize * sizeof(int);
				catalog.Get<ParticleGridNextBuffer>().byteSize = maxParticles * sizeof(int);
			}

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			// ParticleBuffer/both grid buffers are all catalog-converted -- no .Desc() needed,
			// see passes/particle/resources.hpp.
			r.Read<ParticleBuffer>();
			r.Modify<ParticleGridHeadsBuffer>();
			r.Create<ParticleGridNextBuffer>();
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

			ParticleGridBuildPushConstants push{
				.maxParticles = maxParticles,
				.gridSize = gridSize,
				.cellSize = cellSize,
			};

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
			std::uint32_t     groupCount = (maxParticles + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(ParticleGridBuildNode);

} // namespace brassica
