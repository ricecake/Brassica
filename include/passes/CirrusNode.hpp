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

	class ShaderWatcher;

	// Forward-declared only: Before<CloudRenderNode>()/After<WaterNode>() below need just the
	// type's name -- see ClusterLightAssignmentNode.hpp's identical reasoning.
	struct CloudRenderNode;
	struct WaterNode;

	struct CirrusPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f};
		float         worldScale{1.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t transmittanceIndex{0};
		std::uint32_t skyViewIndex{0};
		float         cirrusAlt{10.0f};
		float         cirrusOpacity{0.0125f};
		std::uint32_t weatherBiomeIndex{0};
	};

	// Renders a thin, lower atmospheric cirrus cloud layer at ~10km altitude.
	// Serves as a weather indicator and daytime skybox layer, blending over HdrColor
	// with dual-sided visibility and camera proximity transparency fading.
	// S4: no kPhase -- After<WaterNode>()/Before<CloudRenderNode>() (below) place it in the
	// HdrColor writer chain as real edges instead, immediately after WaterNode so water surfaces
	// do not cover the cirrus cloud layer.
	struct CirrusNode: render::NodeRegistrar<CirrusNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader; // shaders/atmosphere/sky.vert
		FragmentShader           fragShader; // shaders/atmosphere/cirrus.frag
		CirrusPushConstants      push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/atmosphere/cirrus.frag")) {
				spdlog::critical("CirrusNode shader compilation failed.");
				throw std::runtime_error("CirrusNode shader compilation failed.");
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
			push.worldScale = p.worldScale;
			push.cirrusOpacity = p.cirrusOpacity;
		}

		// No Read<GBufferNormal/Material/Depth>() -- same lesson #2 over-claim as
		// AtmosphereCompositeNode.hpp's identical fix. Read<TerrainWeatherBiomeTexture>()
		// restored (S4): catalog-converted since batch 5, so the lesson #10 hazard that
		// justified omitting it is gone. Before<CloudRenderNode>() reproduces the schedule
		// order S0 recorded (Cirrus before Cloud) for these two same-phase HdrColor writers.
		// After<WaterNode>() (S4) is the other half of WaterNode's new Before<CirrusNode>() --
		// see VolumetricRenderNode.hpp's identical comment on why this chain needs real edges.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<TransmittanceLUT>();
			r.Read<SkyViewLUT>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Modify<HdrColor>();
			r.Before<CloudRenderNode>();
			r.After<WaterNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();

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

	BRASSICA_REGISTER_NODE(CirrusNode);

} // namespace brassica
