#pragma once

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPass.hpp"
#include "passes/ResourceKeys.hpp"
#include "Shader.hpp"
#include "vk_mem_alloc.h"

namespace brassica {

	class ShaderWatcher;

	struct TerrainPushConstants {
		glm::mat4  viewProj{1.0f};
		glm::vec4  cameraPos{0.0f, 10.0f, 20.0f, 0.5f}; // xyz = camera position, w = baseTexelSize
		glm::uvec4 gridParams{8, 16, 2048, 1088}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::uvec4 lodOffsets0_3{0u};             // Toroidal offsets for LOD 0-3
		glm::uvec4 lodOffsets4_7{0u};             // Toroidal offsets for LOD 4-7
	};

	class TerrainPass: public RenderPass {
	public:
		TerrainPass(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);
		~TerrainPass() override;

		void InitPipeline(
			vk::Instance            instance,
			vk::Device              device,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher = nullptr,
			vk::PipelineCache       pCache = nullptr
		);

		void UpdateClipmapDescriptor(vk::ImageView clipmapImageView, vk::Sampler clipmapSampler);

		void BuildOrUpdateAccelerationStructure(
			VmaAllocator     allocator,
			const glm::vec3& cameraPos,
			float            baseTexelSize,
			uint32_t         numLODs
		);

		[[nodiscard]] vk::AccelerationStructureKHR GetTLAS() const { return tlas; }

		[[nodiscard]] vk::DescriptorSet GetTerrainDescriptorSet() const { return terrainDescriptorSet; }

		[[nodiscard]] const DispatchLoaderDynamic& GetDls() const { return dls; }

	private:
		DispatchLoaderDynamic dls;

		TaskShader     taskShader;
		MeshShader     meshShader;
		FragmentShader fragShader;

		vk::DescriptorSetLayout terrainSet1Layout{nullptr};
		vk::DescriptorPool      terrainDescriptorPool{nullptr};
		vk::DescriptorSet       terrainDescriptorSet{nullptr};

		struct BufferResource {
			vk::Buffer        buffer{nullptr};
			VmaAllocation     allocation{VK_NULL_HANDLE};
			vk::DeviceAddress deviceAddress{0};
		};

		VmaAllocator lastAllocator{VK_NULL_HANDLE};

		// Acceleration structure resources
		BufferResource               aabbBuffer;
		BufferResource               blasBuffer;
		vk::AccelerationStructureKHR blas{nullptr};
		BufferResource               instanceBuffer;
		BufferResource               tlasBuffer;
		vk::AccelerationStructureKHR tlas{nullptr};
		BufferResource               scratchBuffer;
		glm::vec3                    lastASCameraPos{1e9f, 1e9f, 1e9f};

		void DestroyAccelerationStructures();
		void InitPipelineCustom(
			vk::Instance            instance,
			vk::Device              dev,
			vk::DescriptorSetLayout globalSet0Layout,
			ShaderWatcher*          watcher
		);
	};

	// GBuffer textures are registry-managed now (Create<GBufferPosition/Normal/Albedo/Depth>
	// below) -- the old CreateGBufferTextures/DestroyGBufferTextures reallocation-on-resize
	// logic and the posTex/normTex/albTex/depthTex members it maintained are gone; the physical
	// registry's desc-match reuse (PhysicalResourceRegistry::ProvisionTexture) does that job.
	//
	// The TLAS build itself stays fully out-of-band (TerrainPass::BuildOrUpdateAccelerationStructure,
	// unchanged: its own transient command pool/queue, its own throttle, its own synchronous
	// device.waitIdle()) -- Engine calls it directly in DrawFrame, before constructing the Frame,
	// then registers the result into the registry as an Imported AccelerationStructure. This
	// node's Setup() only *declares* that realization; it never performs the build itself. See
	// the AccelerationStructure resource-kind plan for why moving the build itself into a graph
	// node's Execute() was rejected (a real use-after-free risk against in-flight frames).
	struct TerrainNode {
		using Resources = graph::Declares<
			graph::Read<ShadingRateMap>,
			graph::Create<GBufferPosition>,
			graph::Create<GBufferNormal>,
			graph::Create<GBufferAlbedo>,
			graph::Create<GBufferDepth>,
			graph::Create<TerrainTLAS>>;

		TerrainPass*         pass;
		vk::Extent2D         extent;
		vk::DescriptorSet    globalDescriptorSet;
		TerrainPushConstants pushConstants;

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			// clearColor = {0,0,0,0}, not the default opaque black: deferred.frag reads
			// albedo.a < 0.01 as "no terrain rendered at this pixel, show the gradient
			// background instead" (see ResourceRealization::clearColor's comment,
			// Execution.hpp). Applied to all three color targets, matching the pre-migration
			// TerrainPass exactly, even though only albedo's alpha is actually read.
			uint32_t mapWidth = (ctx.width + 15) / 16;
			uint32_t mapHeight = (ctx.height + 15) / 16;

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<ShadingRateMap>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ShadingRateAttachmentDesc(mapWidth, mapHeight, vk::Format::eR8Uint),
				}
			);
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
			// Declared unconditionally every frame, whether or not BuildOrUpdateAccelerationStructure
			// actually rebuilt this frame (it self-throttles by camera movement and Engine calls it
			// unconditionally before this Setup runs) -- simpler than threading a "did it actually
			// rebuild" flag through just to skip an otherwise-harmless redundant barrier.
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<TerrainTLAS>(),
					.access = graph::AccessKind::Write,
					.desc = graph::AccelerationStructureDesc(),
				}
			);
			return r;
		}

		void Execute(graph::CommandBuffer& cmd) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(cmd.vkCmd));
			pass->BindForDraw(vkCmd, extent);

			if (globalDescriptorSet) {
				vkCmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pass->GetPipelineLayout(),
					0,
					globalDescriptorSet,
					nullptr
				);
			}
			vk::DescriptorSet terrainSet = pass->GetTerrainDescriptorSet();
			if (terrainSet) {
				vkCmd.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					pass->GetPipelineLayout(),
					1,
					terrainSet,
					nullptr
				);
			}

			vkCmd.pushConstants(
				pass->GetPipelineLayout(),
				vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
				0,
				sizeof(TerrainPushConstants),
				&pushConstants
			);

			// Dispatch task groups: ceil(totalMeshlets / 32)
			uint32_t taskGroupCount = (pushConstants.gridParams.z + 31) / 32;
			pass->DrawMeshTasksEXT(vkCmd, taskGroupCount, 1, 1, pass->GetDls());
		}
	};

} // namespace brassica
