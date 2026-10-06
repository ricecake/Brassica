#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/Frame.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "particle/IParticleManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

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
	};

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{8192};
		float         deltaTime{0.016f};
		std::uint32_t gridSize{1024};
		float         cellSize{8.0f};
		std::uint32_t enableLights{1};
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

		if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf || !pGridHeadsBuf || !pGridNextBuf)
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

		inline std::array<vk::DescriptorSetLayout, 3>
		ParticleSetLayouts(const graph::NodeContext& ctx, vk::DescriptorSetLayout particleLayout) {
			return {
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleLayout,
			};
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

			if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf || !pGridHeadsBuf || !pGridNextBuf) {
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

		inline void BindParticleSets(
			vk::CommandBuffer         cmd,
			vk::PipelineBindPoint     bindPoint,
			vk::PipelineLayout        layout,
			const graph::NodeContext& ctx,
			vk::DescriptorSet         particleSet
		) {
			std::array<vk::DescriptorSet, 3> sets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet),
				particleSet,
			};
			if (sets[0] && sets[1] && sets[2]) {
				cmd.bindDescriptorSets(bindPoint, layout, 0, sets, nullptr);
			}
		}

	} // namespace detail

	struct ParticleResetNode {
		using Resources = graph::Declares<
			graph::Create<AboveWaterParticleIndirectBuffer>,
			graph::Create<UnderwaterParticleIndirectBuffer>,
			graph::Create<ParticleGridHeadsBuffer>>;
		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		std::uint32_t                   gridSize{1024};
		vk::DescriptorSetLayout         particleSetLayout;
		vk::DescriptorSet               particleSet;
		detail::ParticleDescriptorCache* descriptorCache{nullptr};

		void Init(
			const render::NodeServices&      services,
			vk::DescriptorSetLayout          setLayout,
			vk::DescriptorSet                set,
			detail::ParticleDescriptorCache& sharedCache
		) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			descriptorCache = &sharedCache;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_reset.comp")) {
				spdlog::critical("ParticleResetNode shader compilation failed.");
				throw std::runtime_error("ParticleResetNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
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
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Write,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Write,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleGridHeadsBuffer>(),
						.access = graph::AccessKind::Write,
						.desc = graph::StorageBufferDesc(gridSize * sizeof(int)),
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(*descriptorCache, registry->GetDevice(), particleSet, registry);
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ParticleResetPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eCompute, resolved.layout, ctx, particleSet);

			ParticleResetPushConstants push{.gridSize = gridSize};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(ParticleResetPushConstants),
				&push
			);

			std::uint32_t groupCount = (gridSize + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	struct ParticleLivenessNode {
		detail::ParticleDescriptorCache* descriptorCache{nullptr};
		using Resources = graph::Declares<
			graph::Modify<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Create<AboveWaterParticleAliveBuffer>,
			graph::Modify<AboveWaterParticleIndirectBuffer>,
			graph::Create<UnderwaterParticleAliveBuffer>,
			graph::Modify<UnderwaterParticleIndirectBuffer>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{8192};
		float                    deltaTime{0.016f};
		float                    waterLevel{0.0f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void SetFrameParams(const render::NodeFrameParams& p) { waterLevel = p.waterLevel; }

		void Init(
			const render::NodeServices&      services,
			vk::DescriptorSetLayout          setLayout,
			vk::DescriptorSet                set,
			detail::ParticleDescriptorCache& sharedCache
		) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			descriptorCache = &sharedCache;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_liveness.comp")) {
				spdlog::critical("ParticleLivenessNode shader compilation failed.");
				throw std::runtime_error("ParticleLivenessNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
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
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleBuffer>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleTypeBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Write,
						.desc =
							[&] {
								graph::ResourceDesc d = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t));
								d.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eTransferSrc);
								return d;
							}(),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Write,
						.desc =
							[&] {
								graph::ResourceDesc d = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t));
								d.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eTransferSrc);
								return d;
							}(),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = indirectDesc,
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(*descriptorCache, registry->GetDevice(), particleSet, registry);
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ParticleLivenessPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eCompute, resolved.layout, ctx, particleSet);

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
			};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(ParticleLivenessPushConstants),
				&push
			);

			std::uint32_t groupCount = (maxParticles + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	struct ParticleGridBuildNode {
		detail::ParticleDescriptorCache* descriptorCache{nullptr};
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Modify<ParticleGridHeadsBuffer>,
			graph::Create<ParticleGridNextBuffer>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{8192};
		std::uint32_t            gridSize{1024};
		float                    cellSize{8.0f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(
			const render::NodeServices&      services,
			vk::DescriptorSetLayout          setLayout,
			vk::DescriptorSet                set,
			detail::ParticleDescriptorCache& sharedCache
		) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			descriptorCache = &sharedCache;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_grid_build.comp")) {
				spdlog::critical("ParticleGridBuildNode shader compilation failed.");
				throw std::runtime_error("ParticleGridBuildNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}

			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleGridHeadsBuffer>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::StorageBufferDesc(gridSize * sizeof(int)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleGridNextBuffer>(),
						.access = graph::AccessKind::Write,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(int)),
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(*descriptorCache, registry->GetDevice(), particleSet, registry);
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ParticleGridBuildPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eCompute, resolved.layout, ctx, particleSet);

			ParticleGridBuildPushConstants push{
				.maxParticles = maxParticles,
				.gridSize = gridSize,
				.cellSize = cellSize,
			};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(ParticleGridBuildPushConstants),
				&push
			);

			std::uint32_t groupCount = (maxParticles + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	struct ParticleBehaviorNode {
		detail::ParticleDescriptorCache* descriptorCache{nullptr};
		using Resources = graph::Declares<
			graph::Modify<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>,
			graph::Read<ParticleGridHeadsBuffer>,
			graph::Read<ParticleGridNextBuffer>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{8192};
		std::uint32_t            gridSize{1024};
		float                    cellSize{8.0f};
		float                    deltaTime{0.016f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(
			const render::NodeServices&      services,
			vk::DescriptorSetLayout          setLayout,
			vk::DescriptorSet                set,
			detail::ParticleDescriptorCache& sharedCache
		) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			descriptorCache = &sharedCache;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/particle_behavior.comp")) {
				spdlog::critical("ParticleBehaviorNode shader compilation failed.");
				throw std::runtime_error("ParticleBehaviorNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
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
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleBuffer>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleTypeBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleGridHeadsBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(gridSize * sizeof(int)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleGridNextBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(int)),
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(*descriptorCache, registry->GetDevice(), particleSet, registry);
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(ParticleBehaviorPushConstants)}
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eCompute, resolved.layout, ctx, particleSet);

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
			};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(ParticleBehaviorPushConstants),
				&push
			);

			std::uint32_t groupCount = (maxParticles + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	struct UnderwaterParticleRenderNode: render::NodeRegistrar<UnderwaterParticleRenderNode> {
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>,
			graph::Modify<GBufferDepth, 1>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::UnderwaterParticleRender;

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

		graph::Recipe Setup(const graph::FrameContext& ctx) {
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
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Graphics,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleTypeBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferDepth>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<HdrColor>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					}
				}
			};
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
			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, descriptorSet.layout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(ParticleRenderPushConstants)}
			};

			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = depthFormat,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eGraphics, resolved.layout, ctx, descriptorSet.set);

			ParticleRenderPushConstants push{.isUnderwater = 1u};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(ParticleRenderPushConstants),
				&push
			);

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vk::Buffer indirectBuf = indirectBuffer;
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

	BRASSICA_REGISTER_NODE(UnderwaterParticleRenderNode);

	struct AboveWaterParticleRenderNode: render::NodeRegistrar<AboveWaterParticleRenderNode> {
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Modify<GBufferDepth, 2>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::ParticleRender;

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

		graph::Recipe Setup(const graph::FrameContext& ctx) {
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
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Graphics,
				.isActive = active,
				.realizations = {
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<ParticleTypeBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
						.access = graph::AccessKind::Read,
						.desc = indirectDesc,
					},
					graph::ResourceRealization{
						.key = graph::IdOf<GBufferDepth>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
					},
					graph::ResourceRealization{
						.key = graph::IdOf<HdrColor>(),
						.access = graph::AccessKind::ReadWrite,
						.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					}
				}
			};
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
			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, descriptorSet.layout);
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eMeshEXT, 0, sizeof(ParticleRenderPushConstants)}
			};

			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = depthFormat,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eGraphics, resolved.layout, ctx, descriptorSet.set);

			ParticleRenderPushConstants push{.isUnderwater = 0u};
			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(ParticleRenderPushConstants),
				&push
			);

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vk::Buffer indirectBuf = indirectBuffer;
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

	BRASSICA_REGISTER_NODE(AboveWaterParticleRenderNode);

	using ParticleRenderNode = AboveWaterParticleRenderNode;

	using ParticleTypeBufferNode = graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare>;

	using ParticleSystemSpec =
		graph::FrameSpec<ParticleTypeBufferNode, ParticleResetNode, ParticleLivenessNode, ParticleGridBuildNode, ParticleBehaviorNode>;
	using ParticleSystemSubgraph = graph::Subgraph<ParticleSystemSpec>;

	struct ParticleSystemNode: render::NodeRegistrar<ParticleSystemNode> {
		using SubgraphType = ParticleSystemSubgraph;
		using Resources = SubgraphType::Resources;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		SubgraphType            m_subgraph;
		vk::DescriptorSetLayout particleSetLayout{nullptr};
		vk::DescriptorPool      particleDescriptorPool{nullptr};
		vk::DescriptorSet       particleSet{nullptr};

		ParticleTypeBufferNode typeBufferNode{std::vector<ParticleType>{
			ParticleType{
				.color = glm::vec4(1.0f, 0.95f, 0.7f, 0.95f),
				.size = 3.0f,
				.gravityScale = 0.0f,
				.drag = 0.1f
			},
			ParticleType{
				.color = glm::vec4(0.1f, 0.95f, 0.85f, 0.95f),
				.size = 2.0f,
				.gravityScale = 0.0f,
				.drag = 0.1f
			},
			ParticleType{
				.color = glm::vec4(0.9f, 1.0f, 0.3f, 1.0f),
				.size = 1.2f,
				.gravityScale = 0.0f,
				.drag = 0.05f
			},
		}};
		ParticleResetNode      resetNode;
		ParticleLivenessNode   livenessNode;
		ParticleGridBuildNode  gridBuildNode;
		ParticleBehaviorNode   behaviorNode;

		// One cache for the one particleSet all four nodes above share -- each node used to own
		// its own ParticleDescriptorCache, which meant whichever of the four ran first in a given
		// frame would vkUpdateDescriptorSets + bind particleSet, and the next one, still thinking
		// *its* cache was stale, would vkUpdateDescriptorSets the exact same already-bound set
		// again -- a real "VkDescriptorSet ... was destroyed or updated" validation error, not a
		// hypothetical one. One shared cache instance means only the first node of the frame to
		// notice a real buffer change actually issues the update; the rest see it's already current.
		detail::ParticleDescriptorCache sharedDescriptorCache{};

		void SetFrameParams(const render::NodeFrameParams& p) { livenessNode.SetFrameParams(p); }

		void Init(const render::NodeServices& services) {
			vk::Device device = services.device;

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
			particleSetLayout = device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 32}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(4);
			particleDescriptorPool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(particleDescriptorPool);
			allocInfo.setSetLayouts(particleSetLayout);
			particleSet = device.allocateDescriptorSets(allocInfo).front();

			resetNode.Init(services, particleSetLayout, particleSet, sharedDescriptorCache);
			livenessNode.Init(services, particleSetLayout, particleSet, sharedDescriptorCache);
			gridBuildNode.Init(services, particleSetLayout, particleSet, sharedDescriptorCache);
			behaviorNode.Init(services, particleSetLayout, particleSet, sharedDescriptorCache);

			auto& inner = m_subgraph.InnerGraph();
			inner.RegisterRef(typeBufferNode);
			inner.RegisterRef(resetNode);
			inner.RegisterRef(livenessNode);
			inner.RegisterRef(gridBuildNode);
			inner.RegisterRef(behaviorNode);
		}

		void Destroy(vk::Device device) {
			resetNode.Destroy(device);
			livenessNode.Destroy(device);
			gridBuildNode.Destroy(device);
			behaviorNode.Destroy(device);

			if (particleDescriptorPool) {
				device.destroyDescriptorPool(particleDescriptorPool);
				particleDescriptorPool = nullptr;
			}
			if (particleSetLayout) {
				device.destroyDescriptorSetLayout(particleSetLayout);
				particleSetLayout = nullptr;
			}
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) { return m_subgraph.Setup(ctx); }

		void Execute(graph::NodeContext& ctx) { m_subgraph.Execute(ctx); }

		[[nodiscard]] graph::Graph& InnerGraph() { return m_subgraph.InnerGraph(); }

		[[nodiscard]] const graph::Graph& InnerGraph() const { return m_subgraph.InnerGraph(); }
	};

	BRASSICA_REGISTER_NODE(ParticleSystemNode);

} // namespace brassica

namespace brassica::graph {
	template <>
	struct NodeKindOfT<brassica::ParticleSystemNode> {
		static constexpr NodeKind value = NodeKind::Subgraph;
	};
} // namespace brassica::graph
