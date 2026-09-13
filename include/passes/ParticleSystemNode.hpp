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
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "VulkanCompat.hpp"

#include "types/Particle.hpp"

namespace brassica {

	struct ParticleLivenessPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
	};

	struct ParticleBehaviorPushConstants {
		std::uint32_t maxParticles{1024};
		float         deltaTime{0.016f};
	};

	struct ParticleResetNode {
		using Resources = graph::Declares<graph::Modify<ParticleIndirectBuffer>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(
			vk::Device               device,
			render::PipelineLibrary* library,
			vk::DescriptorSetLayout  setLayout,
			vk::DescriptorSet        set,
			ShaderWatcher*           watcher = nullptr
		) {
			pipelineLibrary = library;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(device, "shaders/particle_reset.comp");
			if (watcher) {
				watcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
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
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleSetLayout
			};

			render::ComputePipelineRequest request{
				.shader = &compShader,
				.setLayouts = setLayouts,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, globalSet, nullptr);
			}
			if (particleSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 1, particleSet, nullptr);
			}

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

		void Init(
			vk::Device               device,
			render::PipelineLibrary* library,
			vk::DescriptorSetLayout  setLayout,
			vk::DescriptorSet        set,
			ShaderWatcher*           watcher = nullptr
		) {
			pipelineLibrary = library;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(device, "shaders/particle_liveness.comp");
			if (watcher) {
				watcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

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
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleSetLayout
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
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

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, globalSet, nullptr);
			}
			if (particleSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 1, particleSet, nullptr);
			}

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

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;
		std::uint32_t            maxParticles{1024};
		float                    deltaTime{0.016f};
		vk::DescriptorSetLayout  particleSetLayout;
		vk::DescriptorSet        particleSet;

		void Init(
			vk::Device               device,
			render::PipelineLibrary* library,
			vk::DescriptorSetLayout  setLayout,
			vk::DescriptorSet        set,
			ShaderWatcher*           watcher = nullptr
		) {
			pipelineLibrary = library;
			particleSetLayout = setLayout;
			particleSet = set;
			compShader.CompileComputeFromFile(device, "shaders/particle_behavior.comp");
			if (watcher) {
				watcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

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
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleSetLayout
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
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

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, globalSet, nullptr);
			}
			if (particleSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 1, particleSet, nullptr);
			}

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

		void Init(
			vk::Device                   device,
			render::PipelineLibrary*     library,
			const DispatchLoaderDynamic* dispatchLoader,
			vk::Format                   format,
			vk::DescriptorSetLayout      setLayout,
			vk::DescriptorSet            set,
			ShaderWatcher*               watcher = nullptr
		) {
			pipelineLibrary = library;
			dls = dispatchLoader;
			swapchainFormat = format;
			particleSetLayout = setLayout;
			particleSet = set;
			meshShader.CompileMeshFromFile(device, "shaders/particle.mesh");
			fragShader.CompileFragmentFromFile(device, "shaders/particle.frag");
			if (watcher) {
				watcher->RegisterShader(&meshShader);
				watcher->RegisterShader(&fragShader);
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
			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout),
				particleSetLayout
			};
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

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, globalSet, nullptr);
			}
			if (particleSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 1, particleSet, nullptr);
			}

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

	using ParticleSystemSpec = graph::FrameSpec<ParticleResetNode, ParticleLivenessNode, ParticleBehaviorNode, ParticleRenderNode>;
	using ParticleSystemSubgraph = graph::Subgraph<ParticleSystemSpec>;

	struct ParticleSystemNode {
		using SubgraphType = ParticleSystemSubgraph;
		using Resources = SubgraphType::Resources;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		SubgraphType            m_subgraph;
		vk::DescriptorSetLayout particleSetLayout{nullptr};
		vk::DescriptorPool      particleDescriptorPool{nullptr};
		vk::DescriptorSet       particleSet{nullptr};

		ParticleResetNode    resetNode;
		ParticleLivenessNode livenessNode;
		ParticleBehaviorNode behaviorNode;
		ParticleRenderNode   renderNode;

		void Init(
			vk::Device                   device,
			render::PipelineLibrary*     library,
			const DispatchLoaderDynamic* dispatchLoader,
			vk::Format                   format,
			ShaderWatcher*               watcher = nullptr
		) {
			std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
			bindings[0].setBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[1].setBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[2].setBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);
			bindings[3].setBinding(3).setDescriptorType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(1).setStageFlags(vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eMeshEXT);

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

			resetNode.Init(device, library, particleSetLayout, particleSet, watcher);
			livenessNode.Init(device, library, particleSetLayout, particleSet, watcher);
			behaviorNode.Init(device, library, particleSetLayout, particleSet, watcher);
			renderNode.Init(device, library, dispatchLoader, format, particleSetLayout, particleSet, watcher);

			auto& inner = m_subgraph.InnerGraph();
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

		void UpdateDescriptorSet(vk::Device device, const graph::PhysicalResourceRegistry* registry) {
			if (!registry || !particleSet || !device) return;

			auto pBuf = registry->GetBuffer<ParticleBuffer>();
			auto pTypeBuf = registry->GetBuffer<ParticleTypeBuffer>();
			auto pAliveBuf = registry->GetBuffer<ParticleAliveBuffer>();
			auto pIndirectBuf = registry->GetBuffer<ParticleIndirectBuffer>();

			if (!pBuf || !pTypeBuf || !pAliveBuf || !pIndirectBuf) return;

			vk::DescriptorBufferInfo b0{pBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
			vk::DescriptorBufferInfo b1{pTypeBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
			vk::DescriptorBufferInfo b2{pAliveBuf->GetBuffer(), 0, VK_WHOLE_SIZE};
			vk::DescriptorBufferInfo b3{pIndirectBuf->GetBuffer(), 0, VK_WHOLE_SIZE};

			std::array<vk::WriteDescriptorSet, 4> writes{};
			writes[0].setDstSet(particleSet).setDstBinding(0).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b0);
			writes[1].setDstSet(particleSet).setDstBinding(1).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b1);
			writes[2].setDstSet(particleSet).setDstBinding(2).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b2);
			writes[3].setDstSet(particleSet).setDstBinding(3).setDescriptorType(vk::DescriptorType::eStorageBuffer).setBufferInfo(b3);

			device.updateDescriptorSets(writes, nullptr);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			return m_subgraph.Setup(ctx);
		}

		void Execute(graph::NodeContext& ctx) {
			if (ctx.bindless) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.bindless)) {
					UpdateDescriptorSet(registry->GetDevice(), registry);
				}
			}
			m_subgraph.Execute(ctx);
		}

		[[nodiscard]] graph::Graph& InnerGraph() {
			return m_subgraph.InnerGraph();
		}

		[[nodiscard]] const graph::Graph& InnerGraph() const {
			return m_subgraph.InnerGraph();
		}
	};

} // namespace brassica

namespace brassica::graph {
	template <>
	struct NodeKindOfT<brassica::ParticleSystemNode> {
		static constexpr NodeKind value = NodeKind::Subgraph;
	};
} // namespace brassica::graph
