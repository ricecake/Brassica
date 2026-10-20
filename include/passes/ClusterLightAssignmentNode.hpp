#pragma once

#include <array>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "lighting/Light.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct ClusterLightAssignmentNode: render::NodeRegistrar<ClusterLightAssignmentNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!compShader.CompileComputeFromFile(services.device, "shaders/effects/cluster_light_assignment.comp")) {
				spdlog::critical("ClusterLightAssignmentNode shader compilation failed.");
				throw std::runtime_error("ClusterLightAssignmentNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<ClusteredLighting>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&compShader); }

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<ClusteredLighting>();
			// uLights/uLightCount (shaders/lighting.glsl, set 0 binding 2) is now a real graph
			// key (LightsBuffer, ResourceKeys.hpp) -- this Read<>() is a genuine RAW edge against
			// ParticleBehaviorNode's Modify<LightsBuffer>() (its firefly particles write real
			// dynamic lights on top of the CPU's own per-frame write), replacing the manual
			// After<ParticleBehaviorNode>() relation this used to need instead.
			r.Read<LightsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(1, 1, 24);
		}
	};

	BRASSICA_REGISTER_NODE(ClusterLightAssignmentNode);

} // namespace brassica
