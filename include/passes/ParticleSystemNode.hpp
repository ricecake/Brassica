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
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/Particle.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct ParticleLivenessPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
	};

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
	};

	// Unconditionally re-writes the 4 particle SSBO bindings -- no caching here (a function-local
	// static cache would be shared across every call site regardless of which particle system it
	// came from, silently breaking the moment there's more than one).
	// detail::RefreshParticleDescriptorSet (below) is the only caller and owns the actual dedup
	// check before ever getting here, so this is deliberately dumb.
	inline void UpdateParticleDescriptorSet(
		vk::Device                             device,
		vk::DescriptorSet                      particleSet,
		const graph::PhysicalResourceRegistry* registry
	) {
		if (!registry || !particleSet || !device)
			return;

		auto pBuf = registry->GetBuffer<ParticleBuffer>();
		auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
		auto pAliveBuf = registry->GetBuffer<ParticleAliveBuffer>();
		auto pIndirectBuf = registry->GetBuffer<ParticleIndirectBuffer>();

		if (!pBuf || !pTypeBuf || !pAliveBuf || !pIndirectBuf)
			return;

		vk::Buffer b0 = pBuf->GetBuffer();
		vk::Buffer b1 = pTypeBuf->GetBuffer();
		vk::Buffer b2 = pAliveBuf->GetBuffer();
		vk::Buffer b3 = pIndirectBuf->GetBuffer();

		if (!b0 || !b1 || !b2 || !b3)
			return;

		std::array<vk::DescriptorBufferInfo, 4> bufferInfos{
			vk::DescriptorBufferInfo{b0, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b1, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b2, 0, VK_WHOLE_SIZE},
			vk::DescriptorBufferInfo{b3, 0, VK_WHOLE_SIZE}
		};

		std::array<vk::WriteDescriptorSet, 4> writes{};
		for (uint32_t i = 0; i < 4; ++i) {
			writes[i]
				.setDstSet(particleSet)
				.setDstBinding(i)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setBufferInfo(bufferInfos[i]);
		}

		device.updateDescriptorSets(writes, nullptr);
	}

	namespace detail {

		// The 3 descriptor sets every particle node binds, in order: the always-bound frame set
		// (camera/time/frame data), the bindless catalog, and this system's own particle-buffer
		// set. Shared here rather than repeated in each node's Execute -- the four particle nodes
		// otherwise differ only in shader/push-constants/dispatch, not in this boilerplate.
		inline std::array<vk::DescriptorSetLayout, 3>
		ParticleSetLayouts(const graph::NodeContext& ctx, vk::DescriptorSetLayout particleLayout) {
			return {
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleLayout,
			};
		}

		// Tracks the 4 buffer handles UpdateParticleDescriptorSet last wrote into a given set, so
		// RefreshParticleDescriptorSet can skip the vkUpdateDescriptorSets call on every frame
		// where none of them actually changed (the overwhelmingly common case: aliasing is off
		// for these buffers, see PhysicalRegistry::ProvisionBuffer, so a stable ProvisionBuffer
		// result keeps the same handle frame over frame).
		struct ParticleDescriptorCache {
			vk::DescriptorSet         set{nullptr};
			std::array<vk::Buffer, 4> buffers{};
		};

		// Called from ParticleResetNode::Execute specifically, not from ParticleSystemNode::Execute
		// (which is what an earlier version of this code did): ParticleSystemNode is a Subgraph-kind
		// node, and PhysicalExecutionBackend::RunSchedule recurses straight into a Subgraph's inner
		// graph nodes for the real render path (see its own comment, "a backend that recognizes
		// this node as a Subgraph... reads the now-current inner Schedule/Recipes directly rather
		// than going through Execute") -- ParticleSystemNode::Execute (and by extension anything it
		// calls) never actually runs there. ParticleResetNode::Execute does run either way (real
		// backend recursion or Subgraph::Execute's naive fallback), and Reset is Phase::Early --
		// scheduled before Liveness/Behavior/Render every frame -- so refreshing here is exactly
		// once per frame, before anything that reads the set.
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
			auto pAliveBuf = registry->GetBuffer<ParticleAliveBuffer>();
			auto pIndirectBuf = registry->GetBuffer<ParticleIndirectBuffer>();
			if (!pBuf || !pTypeBuf || !pAliveBuf || !pIndirectBuf) {
				return;
			}

			std::array<vk::Buffer, 4> buffers{
				pBuf->GetBuffer(),
				pTypeBuf->GetBuffer(),
				pAliveBuf->GetBuffer(),
				pIndirectBuf->GetBuffer(),
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
		using Resources = graph::Declares<graph::Modify<ParticleIndirectBuffer>>;
		static constexpr graph::Phase kPhase = graph::Phase::Early;

		render::PipelineLibrary*        pipelineLibrary = nullptr;
		ComputeShader                   compShader;
		vk::DescriptorSetLayout         particleSetLayout;
		vk::DescriptorSet               particleSet;
		detail::ParticleDescriptorCache descriptorCache{};

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(services.device, "shaders/particle_reset.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe       r{.domain = graph::ExecutionDomain::Compute};
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleIndirectBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			// Refreshes the particle descriptor set once per frame -- see
			// detail::RefreshParticleDescriptorSet's comment for why this runs here (Reset,
			// Phase::Early) rather than in ParticleSystemNode::Execute.
			if (ctx.bindless) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.bindless)) {
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
		using Resources = graph::Declares<
			graph::Modify<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Modify<ParticleAliveBuffer>,
			graph::Modify<ParticleIndirectBuffer>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{1024};
		float                    deltaTime{0.016f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(services.device, "shaders/particle_liveness.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleTypeBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleAliveBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleIndirectBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = indirectDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			// No descriptor-set update here -- ParticleResetNode::Execute refreshes it once per
			// frame, before this node runs (see detail::RefreshParticleDescriptorSet's comment).
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

			ParticleLivenessPushConstants push{.maxParticles = maxParticles, .deltaTime = deltaTime};
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
			graph::Read<ParticleAliveBuffer>,
			graph::Read<ParticleIndirectBuffer>>;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{1024};
		float                    deltaTime{0.016f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(services.device, "shaders/particle_behavior.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleTypeBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			// No descriptor-set update here -- ParticleResetNode::Execute refreshes it once per
			// frame, before this node runs (see detail::RefreshParticleDescriptorSet's comment).
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

	struct ParticleRenderNode {
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<ParticleAliveBuffer>,
			graph::Read<ParticleIndirectBuffer>,
			graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
			.enableBlend = true,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		const DispatchLoaderDynamic* dls = nullptr;
		vk::Format                   swapchainFormat = vk::Format::eUndefined;
		vk::Format                   depthFormat = vk::Format::eD32Sfloat;
		vk::DescriptorSetLayout      particleSetLayout;
		vk::DescriptorSet            particleSet;
		vk::Buffer                   indirectBuffer;
		std::uint32_t                maxParticles{1024};

		void Init(const render::NodeServices& services, vk::DescriptorSetLayout setLayout, vk::DescriptorSet set) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			swapchainFormat = services.swapchainFormat;
			particleSetLayout = setLayout;
			particleSet = set;
			meshShader.CompileMeshFromFile(services.device, "shaders/particle.mesh");
			fragShader.CompileFragmentFromFile(services.device, "shaders/particle.frag");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&meshShader);
				services.shaderWatcher->RegisterShader(&fragShader);
			}
		}

		void Destroy(vk::Device device) {
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetIndirectBuffer(vk::Buffer buf) { indirectBuffer = buf; }

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(Particle)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleTypeBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(16 * sizeof(ParticleType)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			// No descriptor-set update here -- ParticleResetNode::Execute refreshes it once per
			// frame, before this node runs (see detail::RefreshParticleDescriptorSet's comment).
			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
			render::GraphicsPipelineRequest        request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = depthFormat,
				.setLayouts = setLayouts,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eGraphics, resolved.layout, ctx, particleSet);

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vk::Buffer indirectBuf = indirectBuffer;
			if (!indirectBuf && ctx.bindless) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.bindless)) {
					if (auto physBuf = registry->GetBuffer<ParticleIndirectBuffer>()) {
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

	using ParticleSystemSpec = graph::FrameSpec<
		graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType>,
		ParticleResetNode,
		ParticleLivenessNode,
		ParticleBehaviorNode,
		ParticleRenderNode>;
	using ParticleSystemSubgraph = graph::Subgraph<ParticleSystemSpec>;

	struct ParticleSystemNode: render::NodeRegistrar<ParticleSystemNode> {
		using SubgraphType = ParticleSystemSubgraph;
		using Resources = SubgraphType::Resources;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		SubgraphType            m_subgraph;
		vk::DescriptorSetLayout particleSetLayout{nullptr};
		vk::DescriptorPool      particleDescriptorPool{nullptr};
		vk::DescriptorSet       particleSet{nullptr};

		graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType> typeBufferNode{
			std::vector<ParticleType>(16, ParticleType{})
		};
		ParticleResetNode    resetNode;
		ParticleLivenessNode livenessNode;
		ParticleBehaviorNode behaviorNode;
		ParticleRenderNode   renderNode;

		void Init(const render::NodeServices& services) {
			vk::Device device = services.device;

			std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
			bindings[0]
				.setBinding(0)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[1]
				.setBinding(1)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[2]
				.setBinding(2)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[3]
				.setBinding(3)
				.setDescriptorType(vk::DescriptorType::eStorageBuffer)
				.setDescriptorCount(1)
				.setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);

			vk::DescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.setBindings(bindings);
			particleSetLayout = device.createDescriptorSetLayout(layoutInfo);

			std::array<vk::DescriptorPoolSize, 1> poolSizes{
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 16}
			};
			vk::DescriptorPoolCreateInfo poolInfo{};
			poolInfo.setPoolSizes(poolSizes);
			poolInfo.setMaxSets(4);
			particleDescriptorPool = device.createDescriptorPool(poolInfo);

			vk::DescriptorSetAllocateInfo allocInfo{};
			allocInfo.setDescriptorPool(particleDescriptorPool);
			allocInfo.setSetLayouts(particleSetLayout);
			particleSet = device.allocateDescriptorSets(allocInfo).front();

			resetNode.Init(services, particleSetLayout, particleSet);
			livenessNode.Init(services, particleSetLayout, particleSet);
			behaviorNode.Init(services, particleSetLayout, particleSet);
			renderNode.Init(services, particleSetLayout, particleSet);

			auto& inner = m_subgraph.InnerGraph();
			inner.RegisterRef(typeBufferNode);
			inner.RegisterRef(resetNode);
			inner.RegisterRef(livenessNode);
			inner.RegisterRef(behaviorNode);
			inner.RegisterRef(renderNode);
		}

		void Destroy(vk::Device device) {
			resetNode.Destroy(device);
			livenessNode.Destroy(device);
			behaviorNode.Destroy(device);
			renderNode.Destroy(device);

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

		// The particle descriptor set itself is refreshed from ParticleResetNode::Execute, not
		// here -- this Execute never actually runs on the real backend path at all (see
		// detail::RefreshParticleDescriptorSet's comment), only through Subgraph::Execute's naive
		// fallback, where m_subgraph.Execute below already reaches ParticleResetNode::Execute on
		// its own.
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
