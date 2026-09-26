#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/Frame.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct ParticleLivenessPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
		// Crude, camera-relative depth sort: a particle's own position vs. this decides which
		// alive-list it lands in for this frame (see particle_liveness.comp), not a fixed
		// bird=above/fish=below assumption -- see ParticleLivenessNode::SetFrameParams.
		float waterLevel{0.0f};
	};

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
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

		if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf)
			return;

		vk::Buffer b0 = pBuf->GetBuffer();
		vk::Buffer b1 = pTypeBuf->GetBuffer();
		vk::Buffer b2 = pAboveAliveBuf->GetBuffer();
		vk::Buffer b3 = pAboveIndirectBuf->GetBuffer();
		vk::Buffer b4 = pUnderAliveBuf->GetBuffer();
		vk::Buffer b5 = pUnderIndirectBuf->GetBuffer();

		if (!b0 || !b1 || !b2 || !b3 || !b4 || !b5)
			return;

		std::array<vk::DescriptorBufferInfo, 6> bufferInfos{
			vk::DescriptorBufferInfo{b0, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b1, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b2, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b3, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b4, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b5, 0, VK_WHOLE_SIZE}
		};

		std::array<vk::WriteDescriptorSet, 6> writes{};
		for (uint32_t i = 0; i < 6; ++i) {
			writes[i]
				.setDstSet(particleSet)
				.setDstBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferInfos[i]);
		}

		device.updateDescriptorSets(writes, nullptr);
	}

	namespace detail {

		// One independent 6-binding descriptor set (layout + pool + set), same shape as
		// ParticleSystemNode's own construction below. Used by UnderwaterParticleRenderNode and
		// AboveWaterParticleRenderNode: now that they are independent top-level nodes (promoted out
		// of ParticleSystemNode's Subgraph so their SubPhase::UnderwaterParticleRender/ParticleRender
		// phases actually reach the outer scheduler relative to WaterNode), EngineNodeRegistry::
		// InitAll() initializes them uniformly with no cross-node coordination possible -- each must
		// own its own set pointing at the same underlying buffers, rather than share one set the way
		// the four simulation nodes still do under ParticleSystemNode's single Init(). This also
		// sidesteps the earlier-fixed class of bug (multiple nodes racing to refresh one shared set,
		// UPDATE_AFTER_BIND validation failure) by construction: nothing else ever touches this set.
		struct ParticleDescriptorSet {
			vk::DescriptorSetLayout layout{nullptr};
			vk::DescriptorPool      pool{nullptr};
			vk::DescriptorSet       set{nullptr};
		};

		inline ParticleDescriptorSet CreateParticleDescriptorSet(vk::Device device) {
			ParticleDescriptorSet result;

			std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
			for (uint32_t i = 0; i < 6; ++i) {
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
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 6}
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
			std::array<vk::Buffer, 6> buffers{};
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
			if (!pBuf || !pTypeBuf || !pAboveAliveBuf || !pAboveIndirectBuf || !pUnderAliveBuf || !pUnderIndirectBuf) {
				return;
			}

			std::array<vk::Buffer, 6> buffers{
				pBuf->GetBuffer(),
				pTypeBuf->GetBuffer(),
				pAboveAliveBuf->GetBuffer(),
				pAboveIndirectBuf->GetBuffer(),
				pUnderAliveBuf->GetBuffer(),
				pUnderIndirectBuf->GetBuffer(),
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
		// Create, not Modify: this node's Recipe below realizes both buffers with
		// AccessKind::Write -- a fresh per-frame overwrite (zeroing the indirect draw count), not a
		// read-modify-write of the previous frame's contents. Declaring Modify here (as it used to)
		// made this the *consumer* of a producer that doesn't exist, while ParticleLivenessNode's
		// own Modify<...> of the same keys made it the same thing from the other side -- a genuine
		// mutual dependency the graph correctly rejected as circular. Create<K> instead has an empty
		// Consumes, breaking the cycle: Liveness's Modify<K> now depends on this Create<K> alone.
		using Resources = graph::
			Declares<graph::Create<AboveWaterParticleIndirectBuffer>, graph::Create<UnderwaterParticleIndirectBuffer>>;
		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		vk::DescriptorSetLayout         particleSetLayout;
		vk::DescriptorSet               particleSet;
		detail::ParticleDescriptorCache descriptorCache{};

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
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
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			// eTransferSrc: general-purpose GPU-readback capability (numeric test probes copying
			// the live indirect count to a host-visible buffer), same reasoning as
			// PhysicalResource.hpp's eTransferSrc on every color target -- cheap to carry, not
			// worth a separate preset.
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = true,
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
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					detail::RefreshParticleDescriptorSet(descriptorCache, registry->GetDevice(), particleSet, registry);
				}
			}

			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eCompute, resolved.layout, ctx, particleSet);

			vkCmd.dispatch(1, 1, 1);
		}
	};

	struct ParticleLivenessNode {
		// AliveBuffers are Create, not Modify, for the same reason as ParticleResetNode's indirect
		// buffers above: this node is their sole producer every frame (Recipe realizes both with
		// AccessKind::Write, a fresh overwrite -- liveness is recomputed from ParticleBuffer each
		// frame, not read back from its own prior output), so declaring Modify would demand a
		// producer that doesn't exist. IndirectBuffers stay Modify: these genuinely are read-
		// modify-write, incrementing the count ParticleResetNode's Create<...> just zeroed.
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

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
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
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			// eTransferSrc: general-purpose GPU-readback capability (numeric test probes copying
			// the live indirect count to a host-visible buffer), same reasoning as
			// PhysicalResource.hpp's eTransferSrc on every color target -- cheap to carry, not
			// worth a separate preset.
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = true,
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
						// eTransferSrc: general-purpose GPU-readback capability (numeric test probes
						// copying the live alive-index list to a host-visible buffer), same reasoning
						// as the indirect buffers' own eTransferSrc above.
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

		// No RefreshParticleDescriptorSet call here: ParticleResetNode's Execute already refreshed
		// it for this frame, and is guaranteed to run first (same SubPhase::Prepare, but this
		// node's Modify<...> depends on Reset's Create<...>, so the scheduler orders Reset first
		// regardless). Refreshing again here used to matter, since each node kept its own
		// descriptorCache and couldn't see that a *different* node had already done it this frame
		// -- meaning every one of Reset/Liveness/the two render nodes independently decided "my
		// cache is empty, I must write," calling vkUpdateDescriptorSets on a set the command buffer
		// had already bound moments earlier in the same recording. Vulkan disallows that without
		// UPDATE_AFTER_BIND on the layout, which this one doesn't have -- confirmed as the real
		// cause of the "destroyed or updated without UPDATE_AFTER_BIND" validation error.
		void Execute(graph::NodeContext& ctx) {
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

			ParticleLivenessPushConstants push{
				.maxParticles = maxParticles,
				.deltaTime = deltaTime,
				.waterLevel = waterLevel,
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

	struct ParticleBehaviorNode {
		using Resources = graph::Declares<
			graph::Modify<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>>;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{8192};
		float                    deltaTime{0.016f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
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
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			// eTransferSrc: general-purpose GPU-readback capability (numeric test probes copying
			// the live indirect count to a host-visible buffer), same reasoning as
			// PhysicalResource.hpp's eTransferSrc on every color target -- cheap to carry, not
			// worth a separate preset.
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = true,
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
					}
				}
			};
		}

		void Execute(graph::NodeContext& ctx) {
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

			ParticleBehaviorPushConstants push{.maxParticles = maxParticles, .deltaTime = deltaTime};
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
		// Modify<GBufferDepth, 1>, not plain Modify<GBufferDepth>: this depth-tests particles
		// against (and writes into) the same physical depth image DeferredNode/AtmosphereCompositeNode
		// already read at phase 0/900, but as a distinct, later logical version -- an unversioned
		// Modify here would create a real backward edge to those two earlier, pure-Read consumers
		// ("a later phase cannot satisfy an earlier phase's input"), even though they were never
		// meant to see this write at all. Genuinely later readers (anything past this phase using a
		// plain, unversioned Read<GBufferDepth>) still pick this up automatically via Graph.hpp's
		// version-lifting -- they don't need to know the version number exists.
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>,
			graph::Modify<GBufferDepth, 1>,
			graph::Modify<HdrColor>>;

		// An independent top-level node, not part of ParticleSystemNode's Subgraph: this phase only
		// means anything relative to WaterNode (SubPhase::WaterRender) if the outer scheduler can
		// see it, which a Subgraph child's phase never can (Frame.hpp's Subgraph presents itself to
		// its parent as one opaque node at the *wrapper's* single kPhase, regardless of what phases
		// its own children declare internally).
		static constexpr graph::Phase kPhase = SubPhase::UnderwaterParticleRender;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
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
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			// eTransferSrc: general-purpose GPU-readback capability (numeric test probes copying
			// the live indirect count to a host-visible buffer), same reasoning as
			// PhysicalResource.hpp's eTransferSrc on every color target -- cheap to carry, not
			// worth a separate preset.
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Graphics,
				.isActive = true,
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

		// Refreshes its own descriptor set every frame, unlike the four simulation nodes (which
		// share one set and let only ParticleResetNode refresh it -- see ParticleLivenessNode::
		// Execute's comment for why that coordination is needed there). This node's set is never
		// touched by anything else, so there is no race to coordinate: refreshing unconditionally
		// here is correct and simplest.
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
		// Modify<GBufferDepth, 2>, chaining onto UnderwaterParticleRenderNode's version 1 the same
		// way version 1 chained onto Terrain/EntityNode's version 0 -- see that node's comment for
		// why this needs an explicit version at all instead of a plain Modify<GBufferDepth>.
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Modify<GBufferDepth, 2>,
			graph::Modify<HdrColor>>;

		// See UnderwaterParticleRenderNode's own kPhase comment: an independent top-level node, not
		// part of ParticleSystemNode's Subgraph, is what lets this actually schedule after
		// WaterNode (SubPhase::WaterRender) rather than as its same-phase peer.
		static constexpr graph::Phase kPhase = SubPhase::ParticleRender;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
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
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			// eTransferSrc: general-purpose GPU-readback capability (numeric test probes copying
			// the live indirect count to a host-visible buffer), same reasoning as
			// PhysicalResource.hpp's eTransferSrc on every color target -- cheap to carry, not
			// worth a separate preset.
			indirectDesc.usageMask |= static_cast<std::uint32_t>(
				vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc
			);
			return graph::Recipe{
				.domain = graph::ExecutionDomain::Graphics,
				.isActive = true,
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

		// See UnderwaterParticleRenderNode::Execute's comment: refreshes its own, independent
		// descriptor set unconditionally, since nothing else ever touches it.
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

	// PredefinedBufferNode's phase defaults to Phase::Default (0), which is fine for its other,
	// phase-insensitive users elsewhere in the engine -- but ParticleLivenessNode reads
	// ParticleTypeBuffer at SubPhase::Prepare (-500), and a later phase can never satisfy an
	// earlier one (Graph.hpp's phase check). Explicit SubPhase::Prepare here, not just "earlier
	// than 0": same-phase nodes are still ordered relative to each other by real data dependency,
	// so this Write still sequences before Liveness's Read.
	using ParticleTypeBufferNode = graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType, SubPhase::Prepare>;

	// UnderwaterParticleRenderNode/AboveWaterParticleRenderNode are deliberately NOT part of this
	// spec (or any Subgraph): a Subgraph presents itself to the outer graph as one opaque node at
	// its wrapper's single kPhase (graph/Frame.hpp), so a child's own kPhase only orders it
	// relative to its Subgraph siblings, never relative to an outer sibling like WaterNode. The
	// render nodes need exactly that outer-relative ordering (SubPhase::UnderwaterParticleRender/
	// ParticleRender interleaved with WaterNode's SubPhase::WaterRender), so they are independent,
	// top-level, auto-registered nodes instead -- see their own BRASSICA_REGISTER_NODE calls above.
	// Only the simulation (spawn/liveness/behavior), which nothing outside this system needs to
	// interleave with, stays in this Subgraph.
	using ParticleSystemSpec =
		graph::FrameSpec<ParticleTypeBufferNode, ParticleResetNode, ParticleLivenessNode, ParticleBehaviorNode>;
	using ParticleSystemSubgraph = graph::Subgraph<ParticleSystemSpec>;

	struct ParticleSystemNode: render::NodeRegistrar<ParticleSystemNode> {
		using SubgraphType = ParticleSystemSubgraph;
		using Resources = SubgraphType::Resources;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

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
		}};
		ParticleResetNode      resetNode;
		ParticleLivenessNode   livenessNode;
		ParticleBehaviorNode   behaviorNode;

		// EngineNodeRegistry::SetFrameParamsAll only ever calls SetFrameParams on top-level
		// registered types (this one), never on a Subgraph's inner nodes directly -- so this
		// wrapper has to forward explicitly to whichever inner nodes actually need per-frame data.
		// Only livenessNode does today (waterLevel, for the camera-relative depth sort -- see its
		// own SetFrameParams).
		void SetFrameParams(const render::NodeFrameParams& p) { livenessNode.SetFrameParams(p); }

		// Descriptor set sized for the 4 simulation nodes only now (max 3 concurrently distinct
		// sets rather than the previous 6-node/4-set sizing) -- the two render nodes own their own
		// independent sets (detail::ParticleDescriptorSet) since they are top-level nodes now, see
		// their own Init().
		void Init(const render::NodeServices& services) {
			vk::Device device = services.device;

			std::array<vk::DescriptorSetLayoutBinding, 6> bindings{};
			for (uint32_t i = 0; i < 6; ++i) {
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
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 18}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(3);
			particleDescriptorPool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(particleDescriptorPool);
			allocInfo.setSetLayouts(particleSetLayout);
			particleSet = device.allocateDescriptorSets(allocInfo).front();

			resetNode.Init(services, particleSetLayout, particleSet);
			livenessNode.Init(services, particleSetLayout, particleSet);
			behaviorNode.Init(services, particleSetLayout, particleSet);

			auto& inner = m_subgraph.InnerGraph();
			inner.RegisterRef(typeBufferNode);
			inner.RegisterRef(resetNode);
			inner.RegisterRef(livenessNode);
			inner.RegisterRef(behaviorNode);
		}

		void Destroy(vk::Device device) {
			resetNode.Destroy(device);
			livenessNode.Destroy(device);
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
