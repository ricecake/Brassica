#pragma once

#include <array>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "lighting/Light.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct ClusterLightAssignmentNode : render::NodeRegistrar<ClusterLightAssignmentNode> {
		using Resources = graph::Declares<graph::Create<ClusteredLighting>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            compShader;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cluster_light_assignment.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&compShader);
		}

		void Destroy(vk::Device device) {
			compShader.Destroy(device);
		}

		graph::Recipe Setup(const graph::FrameContext& /*ctx*/) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ClusteredLighting>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc(TOTAL_CLUSTERS * sizeof(ClusterGPU)),
				}
			);
			return r;
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
