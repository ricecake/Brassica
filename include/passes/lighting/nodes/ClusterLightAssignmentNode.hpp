#pragma once

#include <array>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/lighting/Light.hpp"
#include "resources/core.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct ClusterLightAssignmentPushConstants {
		std::uint32_t cpuLightsIndex{0};
		std::uint32_t gpuLightsIndex{0};
	};

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
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&compShader); }

		void Destroy(vk::Device device) { compShader.Destroy(device); }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			// ClusteredLighting is now imported (Engine::DrawFrame's RegisterImportedBuffer) --
			// same reasoning as LightsBuffer's own Modify<>() just below: a real producer-node
			// write against an already-imported key is Modify, not Create.
			r.Modify<ClusteredLighting>();
			// Read<>() of both gives this a real RAW edge against LightManager's CPU upload node
			// and against ParticleBehaviorNode/LightningAgeNode's GPU writes, replacing the
			// manual After<ParticleBehaviorNode>() relation this used to need instead.
			r.Read<CpuLightsBuffer>();
			r.Read<GpuLightsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			ClusterLightAssignmentPushConstants push{
				.cpuLightsIndex = ctx.BufferIndex<CpuLightsBuffer>(),
				.gpuLightsIndex = ctx.BufferIndex<GpuLightsBuffer>(),
			};
			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, compShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch(1, 1, 24);
		}
	};

	BRASSICA_REGISTER_NODE(ClusterLightAssignmentNode);

} // namespace brassica
