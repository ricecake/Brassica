#pragma once

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/SetupBuilder.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "resources/core.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct GpuLightsResetPushConstants {
		std::uint32_t gpuLightsIndex{0};
	};

	// Sole producer (Create<>) of GpuLightsBuffer -- zeroes it fresh every frame before
	// ParticleBehaviorNode's fireflies and LightningAgeNode (both Modify<>()) write into it.
	// Needs its own dispatch, not a self-clear step folded into either of those: a firefly's
	// atomic-claimed slot and some other invocation's unconditional clear of that same slot
	// index have no ordering guarantee relative to each other within one dispatch (unlike the
	// old shared-buffer scheme, where each invocation only ever touched its own idx-derived
	// slot) -- see gpu_lights_reset.comp's own comment. Create-before-accessor (no explicit
	// After<>() needed) orders this before both Modify<>() nodes the same way ParticleResetNode
	// orders ahead of ParticleBehaviorNode/ParticleLivenessNode.
	struct GpuLightsResetNode: render::NodeRegistrar<GpuLightsResetNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/gpu_lights_reset.comp")) {
				spdlog::critical("GpuLightsResetNode shader compilation failed.");
				throw std::runtime_error("GpuLightsResetNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&compShader);
			}
		}

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<GpuLightsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			GpuLightsResetPushConstants push{.gpuLightsIndex = ctx.BufferIndex<GpuLightsBuffer>()};
			render::ResolvedPipeline    resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			constexpr std::uint32_t groupCount =
				(constants::Class::Lighting::MaxGpuLights + 63u) / 64u;
			vkCmd.dispatch(groupCount, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(GpuLightsResetNode);

} // namespace brassica
