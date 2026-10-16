#pragma once

#include <array>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "lighting/Light.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Forward-declared only: After<ParticleBehaviorNode>() below needs just the type's name
	// (TypeName<T>(), ResourceKey.hpp), never a complete type -- including the real definition
	// (passes/ParticleSystemNode.hpp) would pull in the whole particle subsystem for a node that
	// otherwise has nothing to do with it.
	struct ParticleBehaviorNode;

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
			// ParticleBehaviorNode's firefly particles write real dynamic lights into uLights/
			// uLightCount (shaders/particle_behavior.comp, set 0 binding 2 -- the always-bound
			// frame set, not a graph-tracked resource, so there's no key to Read<> here) before
			// this shader reads them. Not yet enforced -- relations are inert until S3's
			// scheduler lands (SetupBuilder.hpp) -- but this makes the real dependency explicit
			// instead of leaving it to an undeclared phase-ordering assumption. If particle
			// nodes are ever moved off Prepare, this is the declaration that needs to start
			// doing real work, not a comment to go find.
			r.After<ParticleBehaviorNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
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

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.dispatch(1, 1, 24);
		}
	};

	BRASSICA_REGISTER_NODE(ClusterLightAssignmentNode);

} // namespace brassica
