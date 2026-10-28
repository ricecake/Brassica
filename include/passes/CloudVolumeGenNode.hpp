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
	// only when forced (hot-reload/debug), mirroring AtmosphereLUTNode's regen-on-change gate.
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

		// TEMPORARY: diagnosing an every-frame redispatch reported from RenderDoc that static
		// code reading couldn't explain (Active-gated Create<> is a documented-working pattern
		// elsewhere, e.g. HostWriteNode). Remove once root-caused.
		std::uint64_t setupCallCount{0};
		std::uint64_t executeCallCount{0};

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			bool active = !hasGenerated || forceRegeneration;
			r.Domain(graph::ExecutionDomain::Compute).Active(active);
			r.Create<CloudDetailVolume>();
			++setupCallCount;
			if (setupCallCount <= 10 || active) {
				spdlog::info(
					"CloudVolumeGenNode::Setup call #{}: hasGenerated={} forceRegeneration={} -> "
					"active={} (executeCallCount so far={})",
					setupCallCount,
					hasGenerated,
					forceRegeneration,
					active,
					executeCallCount
				);
			}
		}

		void Execute(graph::NodeContext& ctx) {
			++executeCallCount;
			spdlog::info(
				"CloudVolumeGenNode::Execute call #{} (hasGenerated was {} on entry)",
				executeCallCount,
				hasGenerated
			);

			push.outStorageIdx = ctx.StorageIndex<CloudDetailVolume>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				spdlog::warn("CloudVolumeGenNode::Execute call #{}: resolved.pipeline was null, skipping dispatch", executeCallCount);
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
