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
#include "VulkanCompat.hpp"

namespace brassica {

	struct WaterPushConstants {
		glm::vec4     cameraPos{0.0f, 10.0f, 20.0f, 0.0f}; // xyz = camera position, w = time
		glm::vec3     waterColor{0.05f, 0.45f, 0.85f};
		float         waterLevel{0.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t padding{0};
	};

	// Authored fresh, not ported from anything -- the acceptance test for the whole Node/Pass
	// unification: no descriptor set, no pipeline layout boilerplate, no positional
	// pipeline-creation call. Just a resource contract, a phase, a pipeline-state literal, and
	// two shaders. Runs at Phase::Late so it composites over whatever DeferredNode already wrote
	// -- both declare a plain Modify<Swapchain> with no version number between them, which is
	// exactly what Phase (rather than a resource-version chain) is for: see Modify's own comment,
	// graph/Declaration.hpp.
	//
	// The first real consumer of GraphicsPipelineState::enableBlend outside a synthetic test --
	// everything ported before this (Gradient/Deferred/Terrain) opaquely overwrites its target,
	// so this is what actually proves the blend-state plumbing works for something real.
	struct WaterNode {
		using Resources = graph::Declares<
			graph::Read<GBufferPosition>,
			graph::Read<GBufferAlbedo>,
			graph::Read<GBufferNormal>,
			graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = graph::Phase::Late;

		// enableShadingRate=false is a real design choice, not just test convenience: this
		// fragment shader is a couple of texture reads and arithmetic, not the kind of expensive
		// per-pixel work VRS exists to coarsen (Terrain's LOD shading, Deferred's ray-query
		// shadows) -- and running it at full resolution avoids any risk of blocky artifacts at a
		// translucent blend boundary. It's also what makes WaterNode the one real, visible node
		// in this migration that's fully testable on a device without mesh-shader/ray-query/VRS
		// support (tests/test_water_node.cpp) rather than only compiling.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableBlend = true,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		MeshShader*                  meshShader = nullptr;
		FragmentShader*              fragShader = nullptr;
		const DispatchLoaderDynamic* dls = nullptr;
		vk::Extent2D                 extent;
		vk::Format                   swapchainFormat = vk::Format::eUndefined;
		WaterPushConstants           push{};

		graph::Recipe Setup(const graph::FrameContext&) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(extent.width, extent.height, swapchainFormat),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();

			std::array<GraphicsShader*, 2>         stages{meshShader, fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 1> setLayouts{static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)};
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(WaterPushConstants)
			}};
			render::GraphicsPipelineRequest        request{
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

			vk::Viewport
				viewport{0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f};
			vkCmd.setViewport(0, viewport);
			vkCmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

			vkCmd.pushConstants(
				resolved.layout,
				vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(WaterPushConstants),
				&push
			);

			if (dls) {
				vkCmd.drawMeshTasksEXT(1, 1, 1, *dls);
			}
		}
	};

} // namespace brassica
