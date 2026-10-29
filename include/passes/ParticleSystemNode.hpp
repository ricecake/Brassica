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
#include "particle/IParticleManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	// Forward-declared only: Before<WaterNode>()/After<WaterNode>() and the Composite-phase
	// writer-chain relations below need just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct WaterNode;
	struct VolumetricRenderNode;
	struct CirrusNode;
	struct CloudCompositeNode;

	struct ParticleResetPushConstants {
		std::uint32_t gridSize{1024};
	};

	struct ParticleGridBuildPushConstants {
		std::uint32_t maxParticles{8192};
		std::uint32_t gridSize{1024};
		float         cellSize{8.0f};
	};

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

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{8192};
		float         deltaTime{0.016f};
		std::uint32_t gridSize{1024};
		float         cellSize{8.0f};
		std::uint32_t enableLights{1};
		// Appended -- same terrain access as ParticleLivenessPushConstants above, for a ground
		// check during particle motion (Stage 3).
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t terrainTextureDim{1024};
		std::uint32_t terrainNumLevels{8};
		float         waterLevel{0.0f};
	};

	struct ParticleRenderPushConstants {
		std::uint32_t isUnderwater{0};
	};

	inline void UpdateParticleDescriptorSet(
		vk::Device                             device,
		vk::DescriptorSet                      particleSet,
		const graph::PhysicalResourceRegistry* registry
	) {
		if (!registry || !particleSet || !device)
			return;

		auto pBuf = registry->GetBuffer<ParticleBuffer>();
		auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
		auto pAboveAliveBuf = registry->GetBuffer<AboveWaterParticleAliveBuffer>();
		auto pAboveIndirectBuf = registry->GetBuffer<AboveWaterParticleIndirectBuffer>();
		auto pUnderAliveBuf = registry->GetBuffer<UnderwaterParticleAliveBuffer>();
		auto pUnderIndirectBuf = registry->GetBuffer<UnderwaterParticleIndirectBuffer>();
		auto pGridHeadsBuf = registry->GetBuffer<ParticleGridHeadsBuffer>();
		auto pGridNextBuf = registry->GetBuffer<ParticleGridNextBuffer>();

		if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf ||
		    !pGridHeadsBuf || !pGridNextBuf)
			return;

		vk::Buffer b0 = pBuf->GetBuffer();
		vk::Buffer b1 = pTypeBuf->GetBuffer();
		vk::Buffer b2 = pAboveAliveBuf->GetBuffer();
		vk::Buffer b3 = pAboveIndirectBuf->GetBuffer();
		vk::Buffer b4 = pUnderAliveBuf->GetBuffer();
		vk::Buffer b5 = pUnderIndirectBuf->GetBuffer();
		vk::Buffer b6 = pGridHeadsBuf->GetBuffer();
		vk::Buffer b7 = pGridNextBuf->GetBuffer();

		if (!b0 || !b1 || !b2 || !b3 || !b4 || !b5 || !b6 || !b7)
			return;

		std::array<vk::DescriptorBufferInfo, 8> bufferInfos{
			vk::DescriptorBufferInfo{b0, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b1, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b2, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b3, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b4, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b5, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b6, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b7, 0, VK_WHOLE_SIZE}
		};

		std::array<vk::WriteDescriptorSet, 8> writes{};
		for (uint32_t i = 0; i < 8; ++i) {
			writes[i]
				.setDstSet(particleSet)
				.setDstBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferInfos[i]);
		}

		device.updateDescriptorSets(writes, nullptr);
	}

	namespace detail {

		struct ParticleDescriptorSet {
			vk::DescriptorSetLayout layout{nullptr};
			vk::DescriptorPool      pool{nullptr};
			vk::DescriptorSet       set{nullptr};
		};

		inline ParticleDescriptorSet CreateParticleDescriptorSet(vk::Device device) {
			ParticleDescriptorSet result;

			std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
			for (uint32_t i = 0; i < 8; ++i) {
				bindings[i]
					.setBinding(i)
					.setDescriptorType(vk::DescriptorType::eStorageBuffer)
					.setDescriptorCount(1)
					.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			}
			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(bindings);
			result.layout = device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(1);
			result.pool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(result.pool);
			allocInfo.setSetLayouts(result.layout);
			result.set = device.allocateDescriptorSets(allocInfo).front();

			return result;
		}

		inline void DestroyParticleDescriptorSet(vk::Device device, ParticleDescriptorSet& s) {
			if (s.pool) {
				device.destroyDescriptorPool(s.pool);
				s.pool = nullptr;
			}
			if (s.layout) {
				device.destroyDescriptorSetLayout(s.layout);
				s.layout = nullptr;
			}
			s.set = nullptr;
		}

		struct ParticleDescriptorCache {
			vk::DescriptorSet         set{nullptr};
			std::array<vk::Buffer, 8> buffers{};
		};

		inline void RefreshParticleDescriptorSet(
			ParticleDescriptorCache&               cache,
			vk::Device                             device,
			vk::DescriptorSet                      particleSet,
			const graph::PhysicalResourceRegistry* registry
		) {
			if (!registry || !particleSet) {
				return;
			}
			auto pBuf = registry->GetBuffer<ParticleBuffer>();
			auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
			auto pAboveAliveBuf = registry->GetBuffer<AboveWaterParticleAliveBuffer>();
			auto pAboveIndirectBuf = registry->GetBuffer<AboveWaterParticleIndirectBuffer>();
			auto pUnderAliveBuf = registry->GetBuffer<UnderwaterParticleAliveBuffer>();
			auto pUnderIndirectBuf = registry->GetBuffer<UnderwaterParticleIndirectBuffer>();
			auto pGridHeadsBuf = registry->GetBuffer<ParticleGridHeadsBuffer>();
			auto pGridNextBuf = registry->GetBuffer<ParticleGridNextBuffer>();

			if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf ||
			    !pGridHeadsBuf || !pGridNextBuf) {
				return;
			}

			std::array<vk::Buffer, 8> buffers{
				pBuf->GetBuffer(),
				pTypeBuf->GetBuffer(),
				pAboveAliveBuf->GetBuffer(),
				pAboveIndirectBuf->GetBuffer(),
				pUnderAliveBuf->GetBuffer(),
				pUnderIndirectBuf->GetBuffer(),
				pGridHeadsBuf->GetBuffer(),
				pGridNextBuf->GetBuffer(),
			};
			if (cache.set == particleSet && cache.buffers == buffers) {
				return;
			}

			UpdateParticleDescriptorSet(device, particleSet, registry);
			cache.set = particleSet;
			cache.buffers = buffers;
		}

	} // namespace detail

	// S4: no kPhase -- this node is the sole Create<> of all three of its buffers, so
	// Create-before-accessor already orders it before every other toucher (ParticleLivenessNode/
	// ParticleGridBuildNode/ParticleBehaviorNode below) regardless of phase.
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

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			r.Create<AboveWaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Create<UnderwaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Create<ParticleGridHeadsBuffer>().Desc(graph::StorageBufferDesc(gridSize * sizeof(int)));
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

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			graph::ResourceDesc aliveDesc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t));
			aliveDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eTransferSrc);

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			r.Modify<ParticleBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(Particle)));
			// No .Desc() here, deliberately: ParticleTypeBuffer has a real dedicated producer
			// (ParticleTypeBufferNode below) whose own desc must win Provision's schedule-last
			// collapse. A hand-rolled StorageBufferDesc here would disagree on hostAccess
			// (None vs the producer's Staged) and, scheduling after the producer, would win --
			// silently turning the producer's WriteSpan into a no-op (BeginHostWrite returns an
			// empty span for HostAccess::None). An empty builder-default desc is explicitly
			// exempt from that last-writer-wins collapse, so leaving this bare is what makes the
			// real producer's desc actually reach Provision().
			r.Read<ParticleTypeBuffer>();
			r.Create<AboveWaterParticleAliveBuffer>().Desc(aliveDesc);
			r.Modify<AboveWaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Create<UnderwaterParticleAliveBuffer>().Desc(aliveDesc);
			r.Modify<UnderwaterParticleIndirectBuffer>().Desc(indirectDesc);
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

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			r.Read<ParticleBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(Particle)));
			r.Modify<ParticleGridHeadsBuffer>().Desc(graph::StorageBufferDesc(gridSize * sizeof(int)));
			r.Create<ParticleGridNextBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(int)));
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

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);

			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(active);
			r.Modify<ParticleBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(Particle)));
			// No .Desc() here, deliberately: ParticleTypeBuffer has a real dedicated producer
			// (ParticleTypeBufferNode below) whose own desc must win Provision's schedule-last
			// collapse. A hand-rolled StorageBufferDesc here would disagree on hostAccess
			// (None vs the producer's Staged) and, scheduling after the producer, would win --
			// silently turning the producer's WriteSpan into a no-op (BeginHostWrite returns an
			// empty span for HostAccess::None). An empty builder-default desc is explicitly
			// exempt from that last-writer-wins collapse, so leaving this bare is what makes the
			// real producer's desc actually reach Provision().
			r.Read<ParticleTypeBuffer>();
			r.Read<AboveWaterParticleAliveBuffer>().Desc(
				graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t))
			);
			r.Read<AboveWaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Read<UnderwaterParticleAliveBuffer>().Desc(
				graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t))
			);
			r.Read<UnderwaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Read<ParticleGridHeadsBuffer>().Desc(graph::StorageBufferDesc(gridSize * sizeof(int)));
			r.Read<ParticleGridNextBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(int)));
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			// Firefly particles write real dynamic lights into uLights/uLightCount on top of the
			// CPU's own per-frame write (shaders/particle_behavior.comp's LIGHTS_BUFFER_WRITABLE
			// path, set 0 binding 2) -- genuinely ReadWrite: it reads the CPU-seeded uLightCount
			// via atomicMax, and zeroes its own claimed slots' intensity unconditionally before
			// conditionally overwriting them. Replaces the manual After<ParticleBehaviorNode>()
			// relation ClusterLightAssignmentNode used to need for this.
			r.Modify<LightsBuffer>();
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

	struct UnderwaterParticleRenderNode: render::NodeRegistrar<UnderwaterParticleRenderNode> {
		// S3: GBufferDepth's version chain (Modify<GBufferDepth,1> here, Modify<GBufferDepth,2>
		// on AboveWaterParticleRenderNode below) is gone -- version lifting and self-modify
		// chaining, the two mechanisms that used to make that chain meaningful, are both deleted
		// (§F: "the particle GBufferDepth v1/v2 chain becomes the two WaterNode relations").
		// Plain Modify<GBufferDepth>() plus an explicit Before<WaterNode>()/After<WaterNode>()
		// (see Setup() below) says the same thing directly: underwater particles render before
		// water covers them, above-water particles render after.
		//
		// S4: no kPhase -- keeping it alongside Before<WaterNode>() became an *active bug* the
		// moment WaterNode itself dropped kPhase: WaterNode defaulted to Phase::Default (0),
		// which the old phase chain then forced to run BEFORE this node's kPhase=1100, directly
		// contradicting Before<WaterNode>() and producing a real G0 cycle (found empirically via
		// test_particle_system's own "Underwater and AboveWater... schedule before and after
		// WaterNode" test, which was silently broken by that earlier batch -- masked because the
		// whole test binary was already in this sandbox's known-failing bucket for an unrelated
		// device reason, so ctest's coarse pass/fail never flagged it). Removing kPhase here
		// fixes it: the explicit relations plus Create-before-accessor are enough on their own.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.enableBlend = true,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		MeshShader                      meshShader;
		FragmentShader                  fragShader;
		const DispatchLoaderDynamic*    dls = nullptr;
		vk::Format                      swapchainFormat = vk::Format::eUndefined;
		vk::Format                      depthFormat = vk::Format::eD32Sfloat;
		detail::ParticleDescriptorSet   descriptorSet{};
		detail::ParticleDescriptorCache descriptorCache{};
		vk::Buffer                      indirectBuffer;
		std::uint32_t                   maxParticles{8192};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			swapchainFormat = services.swapchainFormat;
			descriptorSet = detail::CreateParticleDescriptorSet(services.device);
			if (!meshShader.CompileMeshFromFile(services.device, "shaders/particle.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/particle.frag")) {
				spdlog::critical("UnderwaterParticleRenderNode shader compilation failed.");
				throw std::runtime_error("UnderwaterParticleRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&meshShader);
				services.shaderWatcher->RegisterShader(&fragShader);
			}
		}

		void Destroy(vk::Device device) {
			meshShader.Destroy(device);
			fragShader.Destroy(device);
			detail::DestroyParticleDescriptorSet(device, descriptorSet);
		}

		void SetIndirectBuffer(vk::Buffer buf) { indirectBuffer = buf; }

		// ParticleBuffer/UnderwaterParticleAliveBuffer/IndirectBuffer aren't catalog-converted,
		// so each keeps the exact desc the old hand-rolled Recipe used (via .Desc()) -- same
		// reasoning as every other not-yet-converted key in this migration: an empty
		// builder-default desc would be a real desc regression, not just a style change.
		// ParticleTypeBuffer is the one exception -- see Read<ParticleTypeBuffer>()'s own
		// comment for why it deliberately has no .Desc() here. GBufferDepth/HdrColor are both
		// catalog-converted; no .Desc() needed either.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);

			r.Domain(graph::ExecutionDomain::Graphics);
			r.Active(active);
			r.Read<ParticleBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(Particle)));
			// No .Desc() here, deliberately: ParticleTypeBuffer has a real dedicated producer
			// (ParticleTypeBufferNode below) whose own desc must win Provision's schedule-last
			// collapse. A hand-rolled StorageBufferDesc here would disagree on hostAccess
			// (None vs the producer's Staged) and, scheduling after the producer, would win --
			// silently turning the producer's WriteSpan into a no-op (BeginHostWrite returns an
			// empty span for HostAccess::None). An empty builder-default desc is explicitly
			// exempt from that last-writer-wins collapse, so leaving this bare is what makes the
			// real producer's desc actually reach Provision().
			r.Read<ParticleTypeBuffer>();
			r.Read<UnderwaterParticleAliveBuffer>().Desc(
				graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t))
			);
			r.Read<UnderwaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Modify<GBufferDepth>();
			r.Modify<HdrColor>();
			r.Before<WaterNode>();
			// Not required by any currently-registered graph (neither this node nor
			// AtmosphereCompositeNode/VolumetricRenderNode is live together today -- an
			// unresolved relation target just becomes an inert virtual marker, see
			// RelationalCompiler.hpp), but real the moment this node and the rest of the
			// HdrColor writer chain are ever registered together: Before<WaterNode>() alone only
			// guarantees "before Water," not "before Water's own predecessors," so without these
			// this node would be an unordered HdrColor writer against both and a hard ambiguity.
			r.After<VolumetricRenderNode>();
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

			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 1> extraLayouts{descriptorSet.layout};
			std::array<vk::DescriptorSet, 1>       extraSets{descriptorSet.set};
			ParticleRenderPushConstants            push{.isUnderwater = 1u};
			render::ResolvedPipeline               resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eMeshEXT,
                depthFormat,
                render::ExtraSets{extraLayouts, extraSets}
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vk::Buffer        indirectBuf = indirectBuffer;
			if (!indirectBuf && ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<UnderwaterParticleIndirectBuffer>()) {
						indirectBuf = physBuf->GetBuffer();
					}
				}
			}

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					0,
					1,
					sizeof(ParticleIndirectCommand)
				);
			}
		}
	};

	struct AboveWaterParticleRenderNode: render::NodeRegistrar<AboveWaterParticleRenderNode> {
		// S3: see UnderwaterParticleRenderNode's identical comment -- same GBufferDepth version
		// chain, same fix (plain Modify<GBufferDepth>() + After<WaterNode>() instead of
		// Modify<GBufferDepth,2>). S4: no kPhase -- see UnderwaterParticleRenderNode's identical
		// comment on why keeping it became an active bug, not just redundant.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.enableBlend = true,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		MeshShader                      meshShader;
		FragmentShader                  fragShader;
		const DispatchLoaderDynamic*    dls = nullptr;
		vk::Format                      swapchainFormat = vk::Format::eUndefined;
		vk::Format                      depthFormat = vk::Format::eD32Sfloat;
		detail::ParticleDescriptorSet   descriptorSet{};
		detail::ParticleDescriptorCache descriptorCache{};
		vk::Buffer                      indirectBuffer;
		std::uint32_t                   maxParticles{8192};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			swapchainFormat = services.swapchainFormat;
			descriptorSet = detail::CreateParticleDescriptorSet(services.device);
			if (!meshShader.CompileMeshFromFile(services.device, "shaders/particle.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/particle.frag")) {
				spdlog::critical("AboveWaterParticleRenderNode shader compilation failed.");
				throw std::runtime_error("AboveWaterParticleRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&meshShader);
				services.shaderWatcher->RegisterShader(&fragShader);
			}
		}

		void Destroy(vk::Device device) {
			meshShader.Destroy(device);
			fragShader.Destroy(device);
			detail::DestroyParticleDescriptorSet(device, descriptorSet);
		}

		void SetIndirectBuffer(vk::Buffer buf) { indirectBuffer = buf; }

		// Same reasoning as UnderwaterParticleRenderNode's identical Setup().
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}

			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);

			r.Domain(graph::ExecutionDomain::Graphics);
			r.Active(active);
			r.Read<ParticleBuffer>().Desc(graph::StorageBufferDesc(maxParticles * sizeof(Particle)));
			// No .Desc() here, deliberately: ParticleTypeBuffer has a real dedicated producer
			// (ParticleTypeBufferNode below) whose own desc must win Provision's schedule-last
			// collapse. A hand-rolled StorageBufferDesc here would disagree on hostAccess
			// (None vs the producer's Staged) and, scheduling after the producer, would win --
			// silently turning the producer's WriteSpan into a no-op (BeginHostWrite returns an
			// empty span for HostAccess::None). An empty builder-default desc is explicitly
			// exempt from that last-writer-wins collapse, so leaving this bare is what makes the
			// real producer's desc actually reach Provision().
			r.Read<ParticleTypeBuffer>();
			r.Read<AboveWaterParticleAliveBuffer>().Desc(
				graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t))
			);
			r.Read<AboveWaterParticleIndirectBuffer>().Desc(indirectDesc);
			r.Modify<GBufferDepth>();
			r.Modify<HdrColor>();
			r.After<WaterNode>();
			// Not required by any currently-registered graph -- see
			// UnderwaterParticleRenderNode's identical comment on After<AtmosphereCompositeNode>()/
			// After<VolumetricRenderNode>() for why this is the other half of the same
			// future-safety fix (CirrusNode/CloudCompositeNode are WaterNode's successors in the
			// HdrColor chain, same as Atmosphere/Volumetric are its predecessors). CloudCompositeNode,
			// not CloudRenderNode -- that's a compute pass now, no longer in this chain at all.
			r.Before<CirrusNode>();
			r.Before<CloudCompositeNode>();
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

			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 1> extraLayouts{descriptorSet.layout};
			std::array<vk::DescriptorSet, 1>       extraSets{descriptorSet.set};
			ParticleRenderPushConstants            push{.isUnderwater = 0u};
			render::ResolvedPipeline               resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eMeshEXT,
                depthFormat,
                render::ExtraSets{extraLayouts, extraSets}
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vk::Buffer        indirectBuf = indirectBuffer;
			if (!indirectBuf && ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<AboveWaterParticleIndirectBuffer>()) {
						indirectBuf = physBuf->GetBuffer();
					}
				}
			}

			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT && indirectBuf) {
				dls->vkCmdDrawMeshTasksIndirectEXT(
					static_cast<VkCommandBuffer>(ctx.cmd.vkCmd),
					static_cast<VkBuffer>(indirectBuf),
					0,
					1,
					sizeof(ParticleIndirectCommand)
				);
			}
		}
	};

	// S6: ParticleSystemNode's Subgraph wrapper (and ParticleSystemSpec/ParticleSystemSubgraph
	// beneath it) is gone -- Reset/Liveness/GridBuild/Behavior above are now independent
	// top-level nodes, same as the two render nodes already were, and the real engine (or a
	// test) registers all five of these plus this seed node directly into one flat Graph. The
	// relation-based scheduler doesn't need an inner-Graph indirection to keep this cluster's
	// internal order correct -- Create-before-accessor on the shared buffers already establishes
	// Reset -> Liveness -> GridBuild -> Behavior with no explicit relation needed (verified via a
	// scratch harness during the split; see project memory).
	//
	// The sole producer of ParticleTypeBuffer -- previously a bare PredefinedBufferNode alias
	// with no owner anywhere in production (the particle cluster's 4 readers above all declared
	// their own Read<ParticleTypeBuffer>(), but nothing ever registered a node to Create<> it,
	// so the real engine's graph was rejected outright the one time the cluster was actually
	// enabled: "nothing created the particle type buffer"). A derived type instead of a bare
	// alias specifically so a default-constructed instance (BRASSICA_REGISTER_NODE's own
	// NodeLifecycle<T>, same as every other flat-registered node here) self-seeds real data --
	// there's no per-type tuning a manager would ever mutate at runtime (unlike MaterialManager's
	// m_materials), so a one-time self-seed is simpler than threading a RegisterTypeBufferNode-
	// style call through Engine.cpp for data that never changes. `using Base::Base;` preserves
	// PredefinedBufferNode's own (initialData, desc) constructor for callers (tests) that want to
	// seed different data explicitly.
	//
	// Every entry is deliberately all-zero, not ParticleType's own non-zero in-class defaults:
	// particle.mesh's own fallback logic (`pType.color.a > 0.0 ? pType.color : p.color`,
	// `pType.size > 0.0 ? pType.size : <per-type hardcoded size>`) only kicks in when this
	// buffer's entry is zero -- a non-zero default here would silently override the already-
	// correct per-type color/size the liveness shader assigns per particle on respawn.
	// gravityScale/drag are declared in every particle shader's ParticleType mirror but never
	// actually indexed anywhere today (confirmed: only particle.mesh ever reads
	// uParticleTypes[p.type] at all), so zero is a correct value for them, not just a safe
	// placeholder.
	struct ParticleTypeBufferNode: graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare> {
		using Base = graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare>;
		using Base::Base;

		ParticleTypeBufferNode():
			Base(std::vector<ParticleType>(16, ParticleType{glm::vec4(0.0f), 0.0f, 0.0f, 0.0f, 0.0f})) {}
	};

	// BRASSICA_REGISTER_NODE(UnderwaterParticleRenderNode);
	// BRASSICA_REGISTER_NODE(AboveWaterParticleRenderNode);
	// BRASSICA_REGISTER_NODE(ParticleResetNode);
	// BRASSICA_REGISTER_NODE(ParticleLivenessNode);
	// BRASSICA_REGISTER_NODE(ParticleGridBuildNode);
	// BRASSICA_REGISTER_NODE(ParticleBehaviorNode);
	// BRASSICA_REGISTER_NODE(ParticleTypeBufferNode);

} // namespace brassica
