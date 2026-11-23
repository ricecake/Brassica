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
#include "resources/core.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{8192};
		float         deltaTime{0.016f};
		std::uint32_t gridSize{1024};
		float         cellSize{8.0f};
		std::uint32_t enableLights{1};
		// Appended -- same terrain access as ParticleLivenessPushConstants, for a ground
		// check during particle motion (Stage 3).
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t terrainTextureDim{1024};
		std::uint32_t terrainNumLevels{8};
		float         waterLevel{0.0f};
		std::uint32_t gpuLightsIndex{0};
	};

	// S4: no kPhase -- every key here except ParticleBuffer has a sole Create<> upstream
	// (ParticleLivenessNode for the Alive buffers, ParticleGridBuildNode/ParticleResetNode for
	// the grid buffers, TerrainGenNode for the Terrain textures), each ordering this node after
	// it via Create-before-accessor. ParticleBuffer itself has no sole creator (both this node
	// and ParticleLivenessNode Modify<> it) -- ordered via ParticleLivenessNode's sole-creator
	// edges on the Alive buffers instead (see its own comment).
	struct ParticleBehaviorNode: render::NodeRegistrar<ParticleBehaviorNode> {
		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		std::uint32_t                   maxParticles{8192};
		std::uint32_t                   gridSize{1024};
		float                           cellSize{8.0f};
		float                           deltaTime{0.016f};
		float                           waterLevel{0.0f};
		std::uint32_t                   terrainTextureDim{1024};
		std::uint32_t                   terrainNumLevels{8};
		detail::ParticleDescriptorSet   descriptorSet{};
		detail::ParticleDescriptorCache descriptorCache{};

		void SetFrameParams(const render::NodeFrameParams& p) {
			waterLevel = p.waterLevel;
			terrainTextureDim = p.terrainGridParams.w;
			terrainNumLevels = p.terrainGridParams.x;
		}

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			descriptorSet = detail::CreateParticleDescriptorSet(services.device);
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_behavior.comp")) {
				spdlog::critical("ParticleBehaviorNode shader compilation failed.");
				throw std::runtime_error("ParticleBehaviorNode shader compilation failed.");
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
				registry->Catalog().Get<ParticleBuffer>().byteSize = maxParticles * sizeof(Particle);
			}

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			// ParticleBuffer/both Alive buffers/both grid buffers are all catalog-converted --
			// no .Desc() needed, see passes/particle/resources.hpp.
			r.Modify<ParticleBuffer>();
			// No .Desc() here, deliberately: ParticleTypeBuffer has a real dedicated producer
			// (ParticleTypeBufferNode) whose own desc must win Provision's schedule-last
			// collapse. A hand-rolled StorageBufferDesc here would disagree on hostAccess
			// (None vs the producer's Staged) and, scheduling after the producer, would win --
			// silently turning the producer's WriteSpan into a no-op (BeginHostWrite returns an
			// empty span for HostAccess::None). An empty builder-default desc is explicitly
			// exempt from that last-writer-wins collapse, so leaving this bare is what makes the
			// real producer's desc actually reach Provision().
			r.Read<ParticleTypeBuffer>();
			r.Read<AboveWaterParticleAliveBuffer>();
			// AboveWater/UnderwaterParticleIndirectBuffer are catalog-converted (S2) -- no
			// .Desc() needed.
			r.Read<AboveWaterParticleIndirectBuffer>();
			r.Read<UnderwaterParticleAliveBuffer>();
			r.Read<UnderwaterParticleIndirectBuffer>();
			r.Read<ParticleGridHeadsBuffer>();
			r.Read<ParticleGridNextBuffer>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			// Firefly particles atomic-claim a slot in GpuLightsBuffer and write real dynamic
			// lights into it (shaders/particle_behavior.comp's GPU_LIGHTS_BUFFER_WRITABLE path,
			// bindless binding 6) -- Modify<>(), not Create<>(): GpuLightsResetNode is the sole
			// Create<>(), ordered ahead of this via the normal Create-before-accessor rule.
			r.Modify<GpuLightsBuffer>();
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

			bool enableLights = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr) {
					enableLights = mgr->GetEnableLights();
				}
			}

			ParticleBehaviorPushConstants push{
				.maxParticles = maxParticles,
				.deltaTime = deltaTime,
				.gridSize = gridSize,
				.cellSize = cellSize,
				.enableLights = enableLights ? 1u : 0u,
				.clipmapIndex = ctx.Index<TerrainClipmapTexture>(),
				.minMaxIndex = ctx.Index<TerrainMinMaxTexture>(),
				.terrainTextureDim = terrainTextureDim,
				.terrainNumLevels = terrainNumLevels,
				.waterLevel = waterLevel,
				.gpuLightsIndex = ctx.BufferIndex<GpuLightsBuffer>(),
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

	BRASSICA_REGISTER_NODE(ParticleBehaviorNode);

} // namespace brassica
