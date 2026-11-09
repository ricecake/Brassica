#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
// #include "passes/ResourceKeys.hpp"
#include "resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "../TerrainManager.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct GroundDebrisPushConstants {
		glm::uvec4 gridParams{16u, 256u, 0u, 1088u}; // x = tilesPerSide, y = totalTiles, z = reserved, w = textureDim
		glm::vec4  cameraPos{0.0f, 0.0f, 0.0f, 0.0f}; // xyz = cam pos, w = camera altitude above ground
		std::uint32_t clipmapIndex{0};
		std::uint32_t minMaxIndex{0};
		float         maxGroundDistance{40.0f};       // max camera altitude above ground to render
		float         tileSize{4.0f};                 // world size per tile
		float         heightOffset{0.04f};            // height offset slightly above terrain
		float         time{0.0f};
		std::uint32_t reserved0{0};
		std::uint32_t reserved1{0};
	};

	static_assert(sizeof(GroundDebrisPushConstants) == 64, "GroundDebrisPushConstants size must be 64 bytes");

	struct GroundDebrisNode: render::NodeRegistrar<GroundDebrisNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*               pipelineLibrary = nullptr;
		TaskShader                             taskShader;
		MeshShader                             meshShader;
		FragmentShader                         fragShader;
		const brassica::DispatchLoaderDynamic* dls = nullptr;
		GroundDebrisPushConstants              push{};
		bool                                   active = true;

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/ground_debris.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/ground_debris.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/ground_debris.frag")) {
				spdlog::critical("GroundDebrisNode shader compilation failed.");
				throw std::runtime_error("GroundDebrisNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
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
			push.cameraPos = glm::vec4(p.cameraPosition, 0.0f);
			push.time = p.time;
			push.gridParams.w = p.terrainGridParams.w;

			float groundHeight = 0.0f;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<TerrainManager>()) {
				auto terrain = ServiceLocator::Instance().Get<TerrainManager>();
				groundHeight = terrain->GetCachedGroundHeight(glm::vec2(p.cameraPosition.x, p.cameraPosition.z), 0.0f);
			}

			float altitudeAboveGround = std::max(0.0f, p.cameraPosition.y - groundHeight);
			push.cameraPos.w = altitudeAboveGround;

			// Node is active when camera is within proximity to the ground
			active = (altitudeAboveGround <= push.maxGroundDistance + 10.0f);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics).Active(active);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferMaterial>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();

			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb,
                vk::Format::eR8G8B8A8Unorm
            };

			render::ResolvedPipeline resolved = render::BindGraphics(
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
			uint32_t          taskGroupCount = (push.gridParams.y + 31) / 32;
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(taskGroupCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(GroundDebrisNode);

} // namespace brassica
