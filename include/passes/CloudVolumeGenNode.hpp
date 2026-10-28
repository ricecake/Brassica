#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct CloudVolumeGenPushConstants {
		std::uint32_t outStorageIdx{0xFFFFFFFFu};
		std::uint32_t volumeDim{256};
	};

	// Bakes CloudDetailVolume: a single 256^3 R16 sfloat volume, periodic in all 3 axes over
	// constants::Weather::kCloudVolumePeriod metres. Pure reusable shape/erosion detail -- the
	// weather coverage map (TerrainBiomeNode) carries all the big-picture "where are the clouds"
	// information, so unlike the former 3-cascade system this has no camera-relative state at
	// all and needs no incremental/toroidal update: it's baked once and left alone, re-baked
	// only when forced (hot-reload/debug). The once-only gate lives inside Execute(), not
	// Setup()'s Active() -- see the comment on Setup() for why.
	struct CloudVolumeGenNode: render::NodeRegistrar<CloudVolumeGenNode> {
		render::PipelineLibrary*    pipelineLibrary = nullptr;
		ComputeShader               genShader;
		CloudVolumeGenPushConstants push{};
		bool                        hasGenerated{false};
		bool                        forceRegeneration{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/cloud_volume_gen.comp")) {
				spdlog::critical("CloudVolumeGenNode shader compilation failed.");
				throw std::runtime_error("CloudVolumeGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<CloudDetailVolume>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&genShader); }

		void Destroy(vk::Device device) { genShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { forceRegeneration = p.forceRegeneration; }

		// Stays structurally Active(true) every frame -- NOT gated on hasGenerated/forceRegeneration
		// -- so this node's edges to CloudShadowGenNode/CloudRenderNode are never stripped from
		// the HEFT scheduler's placement graph (Graph.hpp: edges with an inactive endpoint are
		// dropped from both barrier synthesis and ready-time ranking per "§C.4"). Gating via
		// Active(false) instead (what this used to do) left CloudDetailVolume's correctness fine
		// (Lifetime::Persistent, ResourceKeys.hpp, already short-circuits the zero-init path
		// regardless of active status) but silently changed those two downstream nodes' computed
		// ready time, which shifted HEFT's placement and cost ~10x on both -- confirmed on real
		// hardware. The actual once-only bake gate now lives entirely in Execute() below instead.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(true);
			r.Create<CloudDetailVolume>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (hasGenerated && !forceRegeneration) {
				return;
			}

			push.outStorageIdx = ctx.StorageIndex<CloudDetailVolume>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Dispatch 256 / 8 = 32 workgroups per dimension
			vkCmd.dispatch(32, 32, 32);
			hasGenerated = true;
		}
	};

	BRASSICA_REGISTER_NODE(CloudVolumeGenNode);

} // namespace brassica
