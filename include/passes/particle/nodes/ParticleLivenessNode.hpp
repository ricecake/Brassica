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

	struct ParticleLivenessPushConstants {
		std::uint32_t maxParticles{8192};
		float         deltaTime{0.016f};
		float         waterLevel{0.0f};
		std::uint32_t activeParticles{8192};
		std::uint32_t birdCutoff{2785};
		std::uint32_t fishCutoff{5570};
		// Appended, not interleaved -- see feedback_push_constant_struct_editing: new fields go
		// at the end so GLSL/C++ offsets can't silently drift apart from hand-counting mid-struct.
		// Terrain access for occlusion-aware respawn placement (Stage 3) -- wiring only for now.
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t terrainTextureDim{1024};
		std::uint32_t terrainNumLevels{8};
	};

	// S4: no kPhase -- ParticleResetNode's sole Create<> of AboveWater/UnderwaterParticle
	// IndirectBuffer orders it before this node's Modify<> of both via Create-before-accessor;
	// this node is in turn the sole Create<> of AboveWater/UnderwaterParticleAliveBuffer, which
	// orders it before ParticleBehaviorNode's Read<> of both (and transitively establishes
	// Liveness before Behavior for their shared Modify<ParticleBuffer> too -- see
	// ParticleBehaviorNode's comment). Terrain reads are ordered after TerrainGenNode via its
	// own Create-before-accessor, same as everywhere else in the engine.
	struct ParticleLivenessNode: render::NodeRegistrar<ParticleLivenessNode> {
		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		std::uint32_t                   maxParticles{8192};
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
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_liveness.comp")) {
				spdlog::critical("ParticleLivenessNode shader compilation failed.");
				throw std::runtime_error("ParticleLivenessNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
			detail::DestroyParticleDescriptorSet(device, descriptorSet);
		}

		// TerrainClipmap/MinMax/WeatherBiomeTexture reads are all catalog-converted, so the
		// builder's always-pushed (but deliberately empty) realization for each is safe --
		// matching this node's own pre-conversion shape, which declared these three in Resources
		// but never actually realized them (relying on phase ordering + the catalog alone).
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
				catalog.Get<ParticleBuffer>().byteSize = maxParticles * sizeof(Particle);
				const std::uint64_t aliveByteSize = maxParticles * sizeof(std::uint32_t);
				catalog.Get<AboveWaterParticleAliveBuffer>().byteSize = aliveByteSize;
				catalog.Get<UnderwaterParticleAliveBuffer>().byteSize = aliveByteSize;
			}

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			// ParticleBuffer/AboveWaterParticleAliveBuffer/UnderwaterParticleAliveBuffer are all
			// catalog-converted -- no .Desc() needed, see passes/particle/resources.hpp.
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
			r.Create<AboveWaterParticleAliveBuffer>();
			// AboveWater/UnderwaterParticleIndirectBuffer are catalog-converted (S2) -- no
			// .Desc() needed.
			r.Modify<AboveWaterParticleIndirectBuffer>();
			r.Create<UnderwaterParticleAliveBuffer>();
			r.Modify<UnderwaterParticleIndirectBuffer>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<TerrainWeatherBiomeTexture>();
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

			std::uint32_t activeCount = maxParticles;
			float         birdProp = 0.34f;
			float         fishProp = 0.67f;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr) {
					activeCount = std::min(maxParticles, mgr->GetActiveParticles());
					mgr->GetCutoffs(birdProp, fishProp);
				}
			}

			std::uint32_t birdCutoffIdx = static_cast<std::uint32_t>(birdProp * static_cast<float>(activeCount));
			std::uint32_t fishCutoffIdx = static_cast<std::uint32_t>(fishProp * static_cast<float>(activeCount));

			ParticleLivenessPushConstants push{
				.maxParticles = maxParticles,
				.deltaTime = deltaTime,
				.waterLevel = waterLevel,
				.activeParticles = activeCount,
				.birdCutoff = birdCutoffIdx,
				.fishCutoff = fishCutoffIdx,
				.clipmapIndex = ctx.Index<TerrainClipmapTexture>(),
				.minMaxIndex = ctx.Index<TerrainMinMaxTexture>(),
				.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>(),
				.terrainTextureDim = terrainTextureDim,
				.terrainNumLevels = terrainNumLevels,
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

	BRASSICA_REGISTER_NODE(ParticleLivenessNode);

} // namespace brassica
