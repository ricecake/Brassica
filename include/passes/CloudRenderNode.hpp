#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

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

namespace brassica {

	// Forward-declared only: After<CirrusNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct CirrusNode;

	struct alignas(16) CloudRenderPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t transmittanceIndex{0};

		std::uint32_t skyViewIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t cloudShadowIndex{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t volumeIndex{0};
		float         cirrusOpacity{0.0125f};
		std::uint32_t _pad[2]{0};
	};

	// Reads CloudDetailVolume (the single tileable detail-noise volume) and renders it to
	// visible clouds in HdrColor. Minimally populated graphics fragment pass for cloud volume
	// sampling and composition.
	// S4: no kPhase -- After<CirrusNode>() (below) places it in the HdrColor writer chain as a
	// real edge instead.
	struct CloudRenderNode: render::NodeRegistrar<CloudRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader; // shaders/atmosphere/sky.vert
		FragmentShader           fragShader; // shaders/cloud_render.frag
		CloudRenderPushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cloud_render.frag")) {
				spdlog::critical("CloudRenderNode shader compilation failed.");
				throw std::runtime_error("CloudRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<HdrColor>();
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

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			push.sunRadianceAndSkyExp = glm::vec4(p.sunRadiance, p.skyExposure);
		}

		// No Read<GBufferNormal/Material/Depth>() -- same lesson #2 over-claim as
		// AtmosphereCompositeNode.hpp's identical fix. Read<TerrainWeatherBiomeTexture>()
		// restored (S4), same reasoning as CirrusNode.hpp. After<CirrusNode>() is the other
		// half of Cirrus's Before<CloudRenderNode>() -- see its comment.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<CloudDetailVolume>();
			r.Read<TransmittanceLUT>();
			r.Read<SkyViewLUT>();
			r.Read<CloudShadowMap>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Read<TerrainClipmapTexture>();
			r.Modify<HdrColor>();
			r.After<CirrusNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.volumeIndex = ctx.Index<CloudDetailVolume>();
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.cloudShadowIndex = ctx.Index<CloudShadowMap>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();

			std::array<GraphicsShader*, 2> stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			render::ResolvedPipeline       resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eFragment
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(CloudRenderNode);

} // namespace brassica
