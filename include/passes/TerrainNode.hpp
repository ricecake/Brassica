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
#include "terrain/TerrainAccelerationStructure.hpp"

namespace brassica {

	class ShaderWatcher;

	// Mirrors terrain.task/terrain.mesh's shared push_constant block exactly. clipmapIndex is the
	// only field the old TerrainPushConstants didn't have -- the terrain clipmap moved from
	// TerrainPass's own set-1 combined-image-sampler binding to a bindless index here, same
	// change DeferredNode already made. terrain.task doesn't read clipmapIndex (it doesn't sample
	// the clipmap at all), but still shares this struct: a stage only needs to declare the
	// prefix of a push-constant block it actually uses, and every existing field's offset is
	// unchanged since clipmapIndex is strictly appended at the end.
	struct TerrainPushConstants {
		glm::uvec4 gridParams{8, 16, 2048, 1088}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::uvec4 lodOffsets0_3{0u};             // Toroidal offsets for LOD 0-3
		glm::uvec4 lodOffsets4_7{0u};             // Toroidal offsets for LOD 4-7
		std::uint32_t clipmapIndex{0};
	};

	// Replaces TerrainPass: no per-node descriptor set (UpdateClipmapDescriptor and its set-1
	// layout/pool are gone), no push-constant/descriptor mismatch between task and mesh stages --
	// every sampled input is a bindless index, resolved through NodeContext::Index<K>() below.
	// The BLAS/TLAS build stays entirely out-of-band, unchanged, now owned by
	// TerrainAccelerationStructure (terrain/TerrainAccelerationStructure.hpp) rather than a
	// class that also used to own a pipeline.
	//
	// Like GradientNode/DeferredNode, reconstructed fresh every frame -- pipelineLibrary/
	// taskShader/meshShader/fragShader point at Engine-owned, persistent state, and
	// ResolveCached makes re-resolving the same request every frame a cache hit.
	struct TerrainNode {
		using Resources = graph::Declares<
			graph::Create<GBufferPosition>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferDepth>,
			graph::Create<TerrainTLAS>,
			graph::Read<TerrainClipmapTexture>>;

		// Matches TerrainPass::InitPipeline's old hardcoded state exactly (depth test/write on,
		// eLess, eBack culling). enableShadingRate stays false, matching TerrainPass's existing
		// VRS-ignoring behavior -- the old hand-rolled pipeline never chained the VRS pNext at
		// all, so this is a deliberately pixel-identical port; flipping it to honor the mesh
		// shader's own per-primitive gl_PrimitiveShadingRateEXT writes is a separate,
		// separately-validated follow-up.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*      pipelineLibrary = nullptr;
		TaskShader                    taskShader;
		MeshShader                    meshShader;
		FragmentShader                fragShader;
		TerrainAccelerationStructure* terrainAS = nullptr;
		TerrainPushConstants          push{};

		void Init(
			vk::Device                    device,
			render::PipelineLibrary*      library,
			TerrainAccelerationStructure* as,
			ShaderWatcher*                watcher = nullptr
		) {
			pipelineLibrary = library;
			terrainAS = as;
			taskShader.CompileTaskFromFile(device, "shaders/terrain.task");
			meshShader.CompileMeshFromFile(device, "shaders/terrain.mesh");
			fragShader.CompileFragmentFromFile(device, "shaders/terrain.frag");
			if (watcher) {
				RegisterShaders(*watcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const TerrainPushConstants& p) { push = p; }

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			// clearColor = {0,0,0,0}, not the default opaque black: deferred.frag reads
			// albedo.a < 0.01 as "no terrain rendered at this pixel, show the gradient
			// background instead" (see ResourceRealization::clearColor's comment,
			// Execution.hpp). Applied to all three color targets, matching the pre-migration
			// TerrainPass exactly, even though only albedo's alpha is actually read.
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferPosition>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferNormal>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferAlbedo>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR8G8B8A8Unorm),
					.clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);
			// Declared unconditionally every frame, whether or not BuildOrUpdate actually
			// rebuilt this frame (it self-throttles by camera movement and Engine calls it
			// unconditionally before this Setup runs) -- simpler than threading a "did it
			// actually rebuild" flag through just to skip an otherwise-harmless redundant
			// barrier.
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainTLAS>(),
					.access = graph::AccessKind::Write,
					.desc = graph::AccelerationStructureDesc(),
				}
			);
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR16G16B16A16Sfloat,
				vk::Format::eR8G8B8A8Unorm
			};
			std::array<vk::DescriptorSetLayout, 1> setLayouts{static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)};
			std::array<vk::PushConstantRange, 1>   pushConstantRanges{vk::PushConstantRange{
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(TerrainPushConstants)
			}};
			render::GraphicsPipelineRequest        request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.depthFormat = vk::Format::eD32Sfloat,
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
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(TerrainPushConstants),
				&push
			);

			uint32_t taskGroupCount = (push.gridParams.z + 31) / 32;
			if (terrainAS) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, terrainAS->GetDls());
			}
		}
	};

} // namespace brassica
