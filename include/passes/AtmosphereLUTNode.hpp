#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "types/AtmospherePushConstants.hpp"

namespace brassica {

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

	// Mirrors transmittance_lut.comp's push_constant block exactly. outIndex is declared at an
	// explicit layout(offset=80) on the GLSL side rather than relying on implicit packing to
	// agree with this: AtmospherePushConstants's last named field (hazeHeight) ends at byte 76,
	// but alignas(16) on its vec3 members pulls sizeof(AtmospherePushConstants) up to 80 -- a gap
	// GLSL's own natural packing has no reason to reproduce on its own. The static_asserts below
	// pin both sides so a future change to AtmospherePushConstants's layout fails to compile here
	// instead of silently drifting (this is exactly the class of bug Terrain shipped with once
	// already this migration).
	struct TransmittanceLUTPushConstants {
		AtmospherePushConstants atmosphere;
		std::uint32_t           outIndex{0};
	};

	static_assert(offsetof(TransmittanceLUTPushConstants, outIndex) == 80);

	struct MultiScatteringLUTPushConstants {
		AtmospherePushConstants atmosphere;
		std::uint32_t           outIndex{0};
		std::uint32_t           transmittanceIndex{0};
	};

	static_assert(offsetof(MultiScatteringLUTPushConstants, outIndex) == 80);
	static_assert(offsetof(MultiScatteringLUTPushConstants, transmittanceIndex) == 84);

	// Replaces AtmosphereLUTPass's transmittance half: no per-node descriptor set, no per-frame
	// descriptor writes -- the output LUT is a bindless storage index
	// (NodeContext::StorageIndex<K>()), resolved fresh every Execute. Not wired into the live
	// graph yet -- nothing today consumes either LUT's output; deciding to light-integrate this
	// into DeferredNode is a separate feature decision, not part of this migration. Ported and
	// tested (tests/test_atmosphere_lut.cpp) so it compiles and behaves correctly against the
	// bindless model, ready for whoever wires it up next.
	struct TransmittanceLUTNode {
		using Resources = graph::Declares<graph::Create<TransmittanceLUT>>;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader*               shader = nullptr;
		AtmosphereRegenerationState* throttle = nullptr;
		AtmospherePushConstants      atmosphere{};

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = throttle->ShouldRegenerate(atmosphere)
			};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			TransmittanceLUTPushConstants push{
				.atmosphere = atmosphere,
				.outIndex = ctx.StorageIndex<TransmittanceLUT>(),
			};

			std::array<vk::DescriptorSetLayout, 1> setLayouts{static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)};
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TransmittanceLUTPushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, globalSet, nullptr);
			}

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(TransmittanceLUTPushConstants),
				&push
			);

			vkCmd.dispatch((256 + 7) / 8, (64 + 7) / 8, 1);

			// Runs only when isActive (Graph::Compile culls inactive nodes before Execute), so
			// this unconditionally marks the throttle as caught up -- see
			// AtmosphereRegenerationState::ShouldRegenerate.
			throttle->MarkRegenerated(atmosphere);
		}
	};

	struct MultiScatteringLUTNode {
		using Resources = graph::Declares<graph::Read<TransmittanceLUT>, graph::Create<MultiScatteringLUT>>;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader*               shader = nullptr;
		AtmosphereRegenerationState* throttle = nullptr;
		AtmospherePushConstants      atmosphere{};

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{
				.domain = graph::ExecutionDomain::Compute,
				.isActive = throttle->ShouldRegenerate(atmosphere)
			};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<MultiScatteringLUT>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(32, 32, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			MultiScatteringLUTPushConstants push{
				.atmosphere = atmosphere,
				.outIndex = ctx.StorageIndex<MultiScatteringLUT>(),
				.transmittanceIndex = ctx.Index<TransmittanceLUT>(),
			};

			std::array<vk::DescriptorSetLayout, 1> setLayouts{static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)};
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MultiScatteringLUTPushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, globalSet, nullptr);
			}

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(MultiScatteringLUTPushConstants),
				&push
			);

			// Unlike TransmittanceLUTNode, this doesn't mark the throttle caught up itself --
			// both nodes check the same criterion against the same per-frame atmosphere value,
			// and TransmittanceLUTNode already does it once for both (matching
			// AtmosphereLUTPass's original behavior exactly: only the transmittance half ever
			// called MarkRegenerated).
			vkCmd.dispatch(1, 1, 1);
		}
	};

} // namespace brassica
