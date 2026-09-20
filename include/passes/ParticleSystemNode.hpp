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
		using Resources = graph::Declares<
			graph::Modify<AboveWaterParticleIndirectBuffer>,
			graph::Modify<UnderwaterParticleIndirectBuffer>>;
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
					.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);
			return r;
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
		using Resources = graph::Declares<
			graph::Modify<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Modify<AboveWaterParticleAliveBuffer>,
			graph::Modify<AboveWaterParticleIndirectBuffer>,
			graph::Modify<UnderwaterParticleAliveBuffer>,
			graph::Modify<UnderwaterParticleIndirectBuffer>>;

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
					.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = indirectDesc,
				}
			);
			return r;
		}

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
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>>;

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
					.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			return r;
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

	struct UnderwaterParticleRenderNode {
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<UnderwaterParticleAliveBuffer>,
			graph::Read<UnderwaterParticleIndirectBuffer>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::UnderwaterParticleRender;

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
					.key = graph::IdOf<UnderwaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<UnderwaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
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
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eGraphics, resolved.layout, ctx, particleSet);

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

	struct AboveWaterParticleRenderNode {
		using Resources = graph::Declares<
			graph::Read<ParticleBuffer>,
			graph::Read<ParticleTypeBuffer>,
			graph::Read<AboveWaterParticleAliveBuffer>,
			graph::Read<AboveWaterParticleIndirectBuffer>,
			graph::Modify<HdrColor>>;

		static constexpr graph::Phase kPhase = SubPhase::ParticleRender;

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
					.key = graph::IdOf<AboveWaterParticleAliveBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = graph::StorageBufferDesc(maxParticles * sizeof(std::uint32_t)),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(sizeof(ParticleIndirectCommand));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AboveWaterParticleIndirectBuffer>(),
					.access = graph::AccessKind::Read,
					.desc = indirectDesc,
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<GraphicsShader*, 2>         stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 3> setLayouts = detail::ParticleSetLayouts(ctx, particleSetLayout);
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
			detail::BindParticleSets(vkCmd, vk::PipelineBindPoint::eGraphics, resolved.layout, ctx, particleSet);

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

	using ParticleRenderNode = AboveWaterParticleRenderNode;

	using ParticleSystemSpec = graph::FrameSpec<
		graph::PredefinedBufferNode<ParticleTypeBuffer, ParticleType>,
		ParticleResetNode,
		ParticleLivenessNode,
		ParticleBehaviorNode,
		UnderwaterParticleRenderNode,
		AboveWaterParticleRenderNode>;
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
		ParticleResetNode            resetNode;
		ParticleLivenessNode         livenessNode;
		ParticleBehaviorNode         behaviorNode;
		UnderwaterParticleRenderNode underwaterRenderNode;
		AboveWaterParticleRenderNode renderNode;

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
				vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 24}
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
			underwaterRenderNode.Init(services, particleSetLayout, particleSet);
			renderNode.Init(services, particleSetLayout, particleSet);

			auto& inner = m_subgraph.InnerGraph();
			inner.RegisterRef(typeBufferNode);
			inner.RegisterRef(resetNode);
			inner.RegisterRef(livenessNode);
			inner.RegisterRef(behaviorNode);
			inner.RegisterRef(underwaterRenderNode);
			inner.RegisterRef(renderNode);
		}

		void Destroy(vk::Device device) {
			resetNode.Destroy(device);
			livenessNode.Destroy(device);
			behaviorNode.Destroy(device);
			underwaterRenderNode.Destroy(device);
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
