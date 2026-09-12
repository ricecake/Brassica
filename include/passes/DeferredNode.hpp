#pragma once

#include <array>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	class ShaderWatcher;

	// Mirrors deferred.frag's push_constant block exactly. The first four fields are the same
	// toroidal-clipmap-sampling parameters TerrainNode's TerrainPushConstants carries (needed
	// here for the ray-query shadow march's terrain-height lookups, not for any vertex
	// transform -- deferred.frag has no vertex-stage use for a view-projection matrix, unlike
	// TerrainNode's mesh shader, so this is a dedicated, slimmer struct rather than a reuse of
	// TerrainPushConstants). The trailing six are bindless indices, filled in by
	// DeferredNode::Execute every frame -- everything else is supplied once, at construction.
	struct DeferredPushConstants {
		glm::uvec4 gridParams{8, 16, 2048, 1088}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::uvec4 lodOffsets0_3{0u};
		glm::uvec4 lodOffsets4_7{0u};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t backgroundIndex{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t tlasIndex{0};
	};

	// Replaces DeferredPass: no per-node descriptor set, no per-frame descriptor writes -- every
	// sampled input is a bindless index in the push-constant block, resolved through
	// NodeContext::Index<K>() below. Read<TerrainClipmapTexture> is a declared dependency (see
	// ResourceKeys.hpp), not a raw vk::ImageView/vk::Sampler smuggled in with no graph edge, the
	// way the old DeferredNode carried clipmapImageView/clipmapSampler fields.
	//
	// Like GradientNode, reconstructed fresh every frame -- pipelineLibrary/vertShader/fragShader
	// point at Engine-owned, persistent state (see Engine::pipelineLibrary/deferredVertShader/
	// deferredFragShader), and ResolveCached makes re-resolving the same request every frame a
	// cache hit.
	struct DeferredNode {
		using Resources = graph::Declares<
			graph::Read<GBufferPosition>,
			graph::Read<GBufferNormal>,
			graph::Read<GBufferAlbedo>,
			graph::Read<GradientBackground>,
			graph::Read<TerrainClipmapTexture>,
			graph::Read<TerrainTLAS>,
			graph::Modify<Swapchain>>;

		// cullMode=eNone (fullscreen triangle) and enableBlend=false match
		// GraphicsPipelineState's own defaults already; only cullMode needs overriding.
		// enableShadingRate stays at its default (true), matching DeferredPass's original
		// pipeline exactly.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		vk::Format               swapchainFormat = vk::Format::eUndefined;
		DeferredPushConstants    push{};

		void Init(
			vk::Device               device,
			render::PipelineLibrary* library,
			vk::Format               format,
			ShaderWatcher*           watcher = nullptr
		) {
			pipelineLibrary = library;
			swapchainFormat = format;
			vertShader.CompileVertexFromFile(device, "shaders/deferred.vert");
			fragShader.CompileFragmentFromFile(device, "shaders/deferred.frag");
			if (watcher) {
				RegisterShaders(*watcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&vertShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			vertShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const DeferredPushConstants& p) { push = p; }

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.backgroundIndex = ctx.Index<GradientBackground>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.tlasIndex = ctx.Index<TerrainTLAS>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 1> setLayouts{static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)};
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(DeferredPushConstants)}
			};
			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, resolved.pipeline);
			}

			vk::DescriptorSet globalSet = static_cast<VkDescriptorSet>(ctx.globalSet);
			if (globalSet) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, globalSet, nullptr);
			}

			vk::Extent2D extent{ctx.width, ctx.height};
			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(DeferredPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

} // namespace brassica
