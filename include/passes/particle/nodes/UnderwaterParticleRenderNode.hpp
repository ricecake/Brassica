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

	struct UnderwaterParticleRenderNode: render::NodeRegistrar<UnderwaterParticleRenderNode> {
		// S3: GBufferDepth's version chain (Modify<GBufferDepth,1> here, Modify<GBufferDepth,2>
		// on AboveWaterParticleRenderNode) is gone -- version lifting and self-modify
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

		// ParticleBuffer/UnderwaterParticleAliveBuffer/UnderwaterParticleIndirectBuffer/
		// GBufferDepth/HdrColor are all catalog-converted -- no .Desc() needed, see
		// passes/particle/resources.hpp. ParticleTypeBuffer is the one with a real dedicated
		// producer -- see its own Read<>() comment below for why.
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
			r.Read<UnderwaterParticleAliveBuffer>();
			r.Read<UnderwaterParticleIndirectBuffer>();
			r.Modify<GBufferDepth>();
			r.Modify<HdrColor>();
			// Before<WaterNode>() alone already places this node before WaterNode's own real
			// successors too (WaterNode.Before<CirrusNode>(), CirrusNode.Before<CloudCompositeNode>(),
			// VolumetricRenderNode.After<CloudCompositeNode>()) -- WaterNode→Cirrus→CloudComposite→
			// Volumetric is a real, fully-ordered HdrColor-writer chain now that both the particle
			// cluster and the cloud pipeline are registered together, so an explicit
			// After<VolumetricRenderNode>() here would contradict that transitive order (Volumetric
			// now runs *after* Water, not before it) and turn this into a real scheduling cycle --
			// confirmed via test_rendergraph.cpp's force-all-active compile. Dropped rather than
			// kept as additional "future safety": Before<WaterNode>() is already sufficient to
			// avoid the unordered-HdrColor-writer ambiguity this used to guard against.
			r.Before<WaterNode>();
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

	BRASSICA_REGISTER_NODE(UnderwaterParticleRenderNode);

} // namespace brassica
