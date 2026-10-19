#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

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
#include "types/AtmospherePushConstants.hpp"
#include "types/SkyViewPushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	// Replaces AtmosphereLUTPass::ShouldRegenerate/MarkRegenerated. Both LUTs are only rebuilt
	// when the atmosphere parameters actually change (mirroring TerrainAccelerationStructure's
	// camera-movement throttle) -- a byte comparison is a safe and exact "did anything change"
	// check since AtmospherePushConstants is a tightly packed POD used directly for GPU upload.
	//
	// Owned externally (by whoever constructs these nodes each frame -- Engine, once wired up,
	// or a test) and handed to each node as a pointer, the same pattern as pipelineLibrary: the
	// node itself is reconstructed fresh every frame, so this state must live somewhere that
	// persists across frames or the throttle would never actually throttle anything.
	struct AtmosphereRegenerationState {
		AtmospherePushConstants lastPush{};
		bool                    hasGenerated{false};

		[[nodiscard]] bool ShouldRegenerate(const AtmospherePushConstants& push) const {
			return !hasGenerated || std::memcmp(&push, &lastPush, sizeof(AtmospherePushConstants)) != 0;
		}

		void MarkRegenerated(const AtmospherePushConstants& push) {
			lastPush = push;
			hasGenerated = true;
		}
	};

	// Mirrors transmittance_lut.comp's push_constant block exactly. Tuning data no longer travels
	// through here -- both shaders read it directly from the global AtmosphereUBO
	// (atmosphere/common.glsl, set 0 binding 4) -- so this is just the one index the shader needs
	// per dispatch.
	struct TransmittanceLUTPushConstants {
		std::uint32_t outIndex{0};
	};

	static_assert(offsetof(TransmittanceLUTPushConstants, outIndex) == 0);

	struct MultiScatteringLUTPushConstants {
		std::uint32_t outIndex{0};
		std::uint32_t transmittanceIndex{0};
	};

	static_assert(offsetof(MultiScatteringLUTPushConstants, outIndex) == 0);
	static_assert(offsetof(MultiScatteringLUTPushConstants, transmittanceIndex) == 4);

	// Replaces AtmosphereLUTPass's transmittance half: no per-node descriptor set, no per-frame
	// descriptor writes -- the output LUT is a bindless storage index
	// (NodeContext::StorageIndex<K>()), resolved fresh every Execute. Not wired into the live
	// graph yet -- nothing today consumes either LUT's output; deciding to light-integrate this
	// into DeferredNode is a separate feature decision, not part of this migration. Ported and
	// tested (tests/test_atmosphere_lut.cpp) so it compiles and behaves correctly against the
	// bindless model, ready for whoever wires it up next.
	struct TransmittanceLUTNode: render::NodeRegistrar<TransmittanceLUTNode> {
		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                shader;
		AtmosphereRegenerationState  localThrottle{};
		AtmosphereRegenerationState* throttle = &localThrottle;
		AtmospherePushConstants      atmosphere{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!shader.CompileComputeFromFile(services.device, "shaders/atmosphere/transmittance_lut.comp")) {
				spdlog::critical("TransmittanceLUTNode shader compilation failed.");
				throw std::runtime_error("TransmittanceLUTNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<TransmittanceLUT>();
			}
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { atmosphere = p.atmosphere; }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(throttle->ShouldRegenerate(atmosphere));
			r.Create<TransmittanceLUT>();
		}

		void Execute(graph::NodeContext& ctx) {
			TransmittanceLUTPushConstants push{
				.outIndex = ctx.StorageIndex<TransmittanceLUT>(),
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, shader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((256 + 7) / 8, (64 + 7) / 8, 1);

			// Runs only when isActive (Graph::Compile culls inactive nodes before Execute), so
			// this unconditionally marks the throttle as caught up -- see
			// AtmosphereRegenerationState::ShouldRegenerate.
			throttle->MarkRegenerated(atmosphere);
		}
	};

	BRASSICA_REGISTER_NODE(TransmittanceLUTNode);

	struct MultiScatteringLUTNode: render::NodeRegistrar<MultiScatteringLUTNode> {
		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                shader;
		AtmosphereRegenerationState  localThrottle{};
		AtmosphereRegenerationState* throttle = &localThrottle;
		AtmospherePushConstants      atmosphere{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!shader.CompileComputeFromFile(services.device, "shaders/atmosphere/multiscattering_lut.comp")) {
				spdlog::critical("MultiScatteringLUTNode shader compilation failed.");
				throw std::runtime_error("MultiScatteringLUTNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<MultiScatteringLUT>();
			}
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { atmosphere = p.atmosphere; }

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(throttle->ShouldRegenerate(atmosphere));
			r.Read<TransmittanceLUT>();
			r.Create<MultiScatteringLUT>();
		}

		void Execute(graph::NodeContext& ctx) {
			MultiScatteringLUTPushConstants push{
				.outIndex = ctx.StorageIndex<MultiScatteringLUT>(),
				.transmittanceIndex = ctx.Index<TransmittanceLUT>(),
			};

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, shader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Unlike TransmittanceLUTNode, this doesn't mark the throttle caught up itself --
			// both nodes check the same criterion against the same per-frame atmosphere value,
			// and TransmittanceLUTNode already does it once for both (matching
			// AtmosphereLUTPass's original behavior exactly: only the transmittance half ever
			// called MarkRegenerated).
			vkCmd.dispatch(1, 1, 1);
		}
	};

	BRASSICA_REGISTER_NODE(MultiScatteringLUTNode);

	struct SkyViewLUTNode: render::NodeRegistrar<SkyViewLUTNode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            shader;
		SkyViewPushConstants     push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!shader.CompileComputeFromFile(services.device, "shaders/atmosphere/sky_view_lut.comp")) {
				spdlog::critical("SkyViewLUTNode shader compilation failed.");
				throw std::runtime_error("SkyViewLUTNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<SkyViewLUT>();
			}
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = p.sunDir;
			push.worldScale = p.worldScale;
			push.sunRadiance = p.sunRadiance;
			push.multiScatScale = p.multiScatScale;
			push.moonDir = p.moonDir;
			push.cloudShadowIntensity = p.cloudShadowIntensity;
			push.moonRadiance = p.moonRadiance;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(true);
			r.Read<TransmittanceLUT>();
			r.Read<MultiScatteringLUT>();
			r.Create<SkyViewLUT>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.outIndex = ctx.StorageIndex<SkyViewLUT>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.multiScatteringIndex = ctx.Index<MultiScatteringLUT>();

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, shader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((192 + 7) / 8, (108 + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(SkyViewLUTNode);

} // namespace brassica
