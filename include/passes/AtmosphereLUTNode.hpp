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
		using Resources = graph::Declares<graph::Create<TransmittanceLUT>>;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                shader;
		AtmosphereRegenerationState  localThrottle{};
		AtmosphereRegenerationState* throttle = &localThrottle;
		AtmospherePushConstants      atmosphere{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			shader.CompileComputeFromFile(services.device, "shaders/atmosphere/transmittance_lut.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
			}
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { atmosphere = p.atmosphere; }

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
				.outIndex = ctx.StorageIndex<TransmittanceLUT>(),
			};

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(TransmittanceLUTPushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
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

	BRASSICA_REGISTER_NODE(TransmittanceLUTNode);

	struct MultiScatteringLUTNode: render::NodeRegistrar<MultiScatteringLUTNode> {
		using Resources = graph::Declares<graph::Read<TransmittanceLUT>, graph::Create<MultiScatteringLUT>>;

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                shader;
		AtmosphereRegenerationState  localThrottle{};
		AtmosphereRegenerationState* throttle = &localThrottle;
		AtmospherePushConstants      atmosphere{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			shader.CompileComputeFromFile(services.device, "shaders/atmosphere/multiscattering_lut.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
			}
		}

		void Destroy(vk::Device device) { shader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) { atmosphere = p.atmosphere; }

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
				.outIndex = ctx.StorageIndex<MultiScatteringLUT>(),
				.transmittanceIndex = ctx.Index<TransmittanceLUT>(),
			};

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(MultiScatteringLUTPushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
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

	BRASSICA_REGISTER_NODE(MultiScatteringLUTNode);

	struct SkyViewLUTNode: render::NodeRegistrar<SkyViewLUTNode> {
		using Resources =
			graph::Declares<graph::Read<TransmittanceLUT>, graph::Read<MultiScatteringLUT>, graph::Create<SkyViewLUT>>;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            shader;
		SkyViewPushConstants     push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			shader.CompileComputeFromFile(services.device, "shaders/atmosphere/sky_view_lut.comp");
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&shader);
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

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = true};
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
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(32, 32, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<SkyViewLUT>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.outIndex = ctx.StorageIndex<SkyViewLUT>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.multiScatteringIndex = ctx.Index<MultiScatteringLUT>();

			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(SkyViewPushConstants)}
			};
			render::ComputePipelineRequest request{
				.shader = &shader,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
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

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(SkyViewPushConstants),
				&push
			);

			vkCmd.dispatch((192 + 7) / 8, (108 + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(SkyViewLUTNode);

} // namespace brassica
