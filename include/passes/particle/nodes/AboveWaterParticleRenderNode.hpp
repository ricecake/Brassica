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

		// Same reasoning as UnderwaterParticleRenderNode's identical Setup() -- ParticleBuffer/
		// AboveWaterParticleAliveBuffer/AboveWaterParticleIndirectBuffer are all
		// catalog-converted, no .Desc() needed.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = true;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<IParticleManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IParticleManager>();
				if (mgr && !mgr->IsEnabled()) {
					active = false;
				}
			}

			r.Domain(graph::ExecutionDomain::Graphics);
			r.Active(active);
			r.Read<ParticleBuffer>();
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
			r.Read<AboveWaterParticleIndirectBuffer>();
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

	BRASSICA_REGISTER_NODE(AboveWaterParticleRenderNode);

} // namespace brassica
