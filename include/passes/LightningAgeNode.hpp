#pragma once

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Forward-declared only: After<TerrainBiomeNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct TerrainBiomeNode;

	struct LightningAgePushConstants {
		float deltaTime{0.016f};
	};

	// Ages every shaders/lighting.glsl LightningStrikePool slot and writes its reserved
	// uLights[] entry each frame -- see lightning_age.comp's own header comment for the full
	// mechanism (mirrors particle_behavior.comp's firefly lights). Dispatched before
	// LightningRenderNode's task/mesh draw so that pass sees this frame's already-aged state.
	struct LightningAgeNode: render::NodeRegistrar<LightningAgeNode> {
		render::PipelineLibrary*  pipelineLibrary = nullptr;
		ComputeShader             ageShader;
		LightningAgePushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!ageShader.CompileComputeFromFile(services.device, "shaders/lightning_age.comp")) {
				spdlog::critical("LightningAgeNode shader compilation failed.");
				throw std::runtime_error("LightningAgeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&ageShader);
			}
		}

		void Destroy(vk::Device device) { ageShader.Destroy(device); }

		// push.deltaTime stays at its hardcoded 60fps-assumption default -- matches
		// ParticleSystemNode's own identical approach (its deltaTime member is never updated
		// from SetFrameParams either); no node in this codebase threads a real measured delta
		// through NodeFrameParams today, so this isn't a new imprecision.
		void SetFrameParams(const render::NodeFrameParams&) {}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Modify<LightningStrikePool>();
			r.Modify<LightsBuffer>();
			// climate_ca.comp's trigger-write (TerrainBiomeNode) and this node's aging both
			// Modify<LightningStrikePool>() with no natural read-based ordering between them --
			// After<>() resolves the ambiguity deterministically. Detect first, age second; a
			// freshly-claimed slot aging by one frame's worth of time immediately is harmless.
			r.After<TerrainBiomeNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, ageShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(1, 1, 1); // LIGHTNING_MAX_STRIKES (32) threads, local_size_x = 32
		}
	};

	BRASSICA_REGISTER_NODE(LightningAgeNode);

} // namespace brassica
