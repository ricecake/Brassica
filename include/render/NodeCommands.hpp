#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"

// The Vulkan-aware bind/push counterpart Execution.hpp's NodeContext comment points at: every
// pass's Execute() rebuilds the same {frameSet, globalSet} descriptor-set arrays and a one-range
// push-constant block around a call to PipelineLibrary::ResolveCached, then binds pipeline/sets/
// push constants in the same order. BindCompute/BindGraphics below are that shared shape;
// dispatch dimensions, draw calls, and barriers stay in each node since those are genuinely
// per-pass.

namespace brassica::render {

	// Node-owned descriptor sets beyond the two every node already gets for free (set 0 = frame
	// UBO, set 1 = bindless global). Appended starting at index 2, e.g.
	// HiZDownsampleNode's atomic-counter set. layouts and sets must be the same length.
	struct ExtraSets {
		std::span<const vk::DescriptorSetLayout> layouts{};
		std::span<const vk::DescriptorSet>       sets{};
	};

	namespace detail {

		// 2 (frame + global) plus however many ExtraSets a node adds -- HiZDownsampleNode's
		// counter set is the only caller with any today, so 8 is headroom, not a measured need.
		inline constexpr std::size_t kMaxBoundSets = 8;

		struct CombinedSets {
			std::array<vk::DescriptorSetLayout, kMaxBoundSets> layouts{};
			std::array<vk::DescriptorSet, kMaxBoundSets>       sets{};
			std::uint32_t                                      count = 0;
			bool                                               allValid = false;
		};

		inline CombinedSets CombineSets(const graph::NodeContext& ctx, ExtraSets extra) {
			CombinedSets combined;
			combined.layouts[0] = static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout);
			combined.layouts[1] = static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout);
			combined.sets[0] = static_cast<VkDescriptorSet>(ctx.frameSet);
			combined.sets[1] = static_cast<VkDescriptorSet>(ctx.globalSet);
			combined.count = 2;
			for (std::size_t i = 0; i < extra.layouts.size(); ++i) {
				combined.layouts[combined.count] = extra.layouts[i];
				combined.sets[combined.count] = extra.sets[i];
				++combined.count;
			}
			combined.allValid = true;
			for (std::uint32_t i = 0; i < combined.count; ++i) {
				if (!combined.sets[i]) {
					combined.allValid = false;
				}
			}
			return combined;
		}

	} // namespace detail

	// Overload for a shader with no push constants at all (e.g. ClusterLightAssignmentNode) --
	// everything else is identical to the PushT overload below, just without a push-constant
	// range on the request or a pushConstants call after binding.
	inline ResolvedPipeline
	BindCompute(graph::NodeContext& ctx, PipelineLibrary& library, ComputeShader& shader, ExtraSets extra = {}) {
		detail::CombinedSets sets = detail::CombineSets(ctx, extra);

		ComputePipelineRequest request{
			.shader = &shader,
			.setLayouts = std::span(sets.layouts.data(), sets.count),
		};
		ResolvedPipeline resolved = library.ResolveCached(request);
		if (!resolved.pipeline) {
			return resolved;
		}

		vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
		vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
		if (sets.allValid) {
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute,
				resolved.layout,
				0,
				std::span(sets.sets.data(), sets.count),
				nullptr
			);
		}
		return resolved;
	}

	// Resolves `shader` to a cached compute pipeline, binds it, binds the frame/global (+ extra)
	// descriptor sets, and pushes `push` as the pipeline's sole push-constant range. Everything
	// here is skipped if the pipeline failed to resolve (e.g. mid shader-hot-reload), so the
	// caller only needs to guard its own dispatch call on the returned pipeline being non-null.
	template <typename PushT>
	ResolvedPipeline BindCompute(
		graph::NodeContext& ctx,
		PipelineLibrary&    library,
		ComputeShader&      shader,
		const PushT&        push,
		ExtraSets           extra = {}
	) {
		detail::CombinedSets sets = detail::CombineSets(ctx, extra);

		std::array<vk::PushConstantRange, 1> pushRanges{
			vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushT)}
		};
		ComputePipelineRequest request{
			.shader = &shader,
			.setLayouts = std::span(sets.layouts.data(), sets.count),
			.pushConstantRanges = pushRanges,
		};
		ResolvedPipeline resolved = library.ResolveCached(request);
		if (!resolved.pipeline) {
			return resolved;
		}

		vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
		vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
		if (sets.allValid) {
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute,
				resolved.layout,
				0,
				std::span(sets.sets.data(), sets.count),
				nullptr
			);
		}
		vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushT), &push);
		return resolved;
	}

	// Graphics counterpart. Also sets a viewport/scissor spanning ctx.width x ctx.height, since
	// every current graphics node wants exactly that -- a node needing anything else should bind
	// manually instead of calling this. pushStage is a parameter (not fixed to eFragment) because
	// push constants land in vertex, fragment, or mesh stages depending on the pass.
	template <typename PushT>
	ResolvedPipeline BindGraphics(
		graph::NodeContext&              ctx,
		PipelineLibrary&                 library,
		std::span<GraphicsShader* const> stages,
		const GraphicsPipelineState&     state,
		std::span<const vk::Format>      colorFormats,
		const PushT&                     push,
		vk::ShaderStageFlags             pushStage,
		vk::Format                       depthFormat = vk::Format::eUndefined,
		ExtraSets                        extra = {}
	) {
		detail::CombinedSets sets = detail::CombineSets(ctx, extra);

		std::array<vk::PushConstantRange, 1> pushRanges{vk::PushConstantRange{pushStage, 0, sizeof(PushT)}};
		GraphicsPipelineRequest              request{
						 .stages = stages,
						 .state = state,
						 .colorFormats = colorFormats,
						 .depthFormat = depthFormat,
						 .setLayouts = std::span(sets.layouts.data(), sets.count),
						 .pushConstantRanges = pushRanges,
        };
		ResolvedPipeline resolved = library.ResolveCached(request);
		if (!resolved.pipeline) {
			return resolved;
		}

		vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
		vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
		if (sets.allValid) {
			vkCmd.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				resolved.layout,
				0,
				std::span(sets.sets.data(), sets.count),
				nullptr
			);
		}

		vk::Extent2D extent{ctx.width, ctx.height};
		vk::Viewport
			viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
		vkCmd.setViewport(0, viewport);
		vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

		vkCmd.pushConstants(resolved.layout, pushStage, 0, sizeof(PushT), &push);
		return resolved;
	}

} // namespace brassica::render
