#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	class ShaderWatcher;

	// Forward-declared only: After<DeferredNode>()/Before<CirrusNode>() below need just
	// the type's name -- see ClusterLightAssignmentNode.hpp's identical reasoning.
	struct DeferredNode;
	struct CirrusNode;

	struct WaterPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		glm::vec3     waterColor{0.05f, 0.45f, 0.85f};
		float         waterLevel{0.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t sceneColor{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t gMaterialIndex{0};
		std::uint32_t skyViewIndex{0};
	};

	// Authored fresh, not ported from anything -- the acceptance test for the whole Node/Pass
	// unification: no descriptor set, no pipeline layout boilerplate, no positional
	// pipeline-creation call. Just a resource contract, a phase, a pipeline-state literal, and
	// two shaders.
	//
	// The first real consumer of GraphicsPipelineState::enableBlend outside a synthetic test --
	// everything ported before this (Gradient/Deferred/Terrain) opaquely overwrites its target,
	// so this is what actually proves the blend-state plumbing works for something real.
	struct WaterNode: render::NodeRegistrar<WaterNode> {
		// S4: no kPhase -- After<VolumetricRenderNode>()/Before<CirrusNode>() (below) place it in
		// the HdrColor writer chain as real edges instead. (The old SubPhase::WaterRender's own
		// history -- fixing a prior collision with ParticleSystemNode's Phase::Late -- is moot now
		// that ordering doesn't come from a phase number at all.)
		//
		// enableShadingRate=false is a real design choice, not just test convenience: this
		// fragment shader is a couple of texture reads and arithmetic, not the kind of expensive
		// per-pixel work VRS exists to coarsen (Terrain's LOD shading, Deferred's ray-query
		// shadows) -- and running it at full resolution avoids any risk of blocky artifacts at a
		// translucent blend boundary. It's also what makes WaterNode the one real, visible node
		// in this migration that's fully testable on a device without mesh-shader/ray-query/VRS
		// support (tests/test_water_node.cpp) rather than only compiling.
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		TaskShader                   taskShader;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		const DispatchLoaderDynamic* dls = nullptr;
		vk::Format                   swapchainFormat = vk::Format::eUndefined;
		WaterPushConstants           push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			swapchainFormat = services.swapchainFormat;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/water.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/water.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/water.frag")) {
				spdlog::critical("WaterNode shader compilation failed.");
				throw std::runtime_error("WaterNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<HdrColor>();
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

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.gridParams = p.terrainGridParams;
			push.waterColor = p.waterColor;
			push.waterLevel = p.waterLevel;
		}

		// Unlike AtmosphereCompositeNode/CirrusNode/CloudRenderNode/VolumetricRenderNode above,
		// this one genuinely needs the whole GBuffer group: Position/Albedo/Normal/Material are
		// all indexed in Execute, and Depth is bound as the real depth-test attachment
		// (kPipelineState.depthTest=true) even though it's never indexed directly. All five are
		// Read<>, not Modify<> -- despite depthTest=true, this node never writes depth, so it's
		// not a writer-pairwise participant on any GBuffer key at all.
		//
		// After<VolumetricRenderNode>()/Before<CirrusNode>() (S4) close the two gaps in the
		// HdrColor writer chain that previously relied on phase-int ordering alone -- see
		// VolumetricRenderNode.hpp's identical comment.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<GBufferNormal>();
			r.Read<GBufferMaterial>();
			r.Read<GBufferDepth>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<SkyViewLUT>();
			r.Modify<HdrColor>();
			r.After<DeferredNode>();
			r.Before<CirrusNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.sceneColor = ctx.Index<HdrColor>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.gMaterialIndex = ctx.Index<GBufferMaterial>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			render::ResolvedPipeline       resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT |
                    vk::ShaderStageFlagBits::eFragment,
                vk::Format::eD32Sfloat
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				std::uint32_t taskGroupCount = (push.gridParams.z + 31) / 32;
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(WaterNode);

} // namespace brassica
