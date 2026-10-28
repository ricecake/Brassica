#pragma once

#include <array>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/SkyPushConstants.hpp"

namespace brassica {

	class ShaderWatcher;

	// Draws the sky-view/transmittance LUTs composited with the sun/moon/star discs into
	// AtmosphereRadiance, DeferredNode's backdrop wherever no opaque surface was rendered.
	// Deliberately not named AtmosphereCompositeNode: that name is reserved for the node that fogs
	// already-shaded scene geometry with aerial perspective/underwater extinction (see
	// AtmosphereCompositeNode.hpp) -- a distinct concept from drawing the sky background, even
	// though this node was briefly (and incorrectly) named that.
	struct SkyBackgroundNode: render::NodeRegistrar<SkyBackgroundNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		SkyPushConstants         push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/atmosphere/sky.frag")) {
				spdlog::critical("SkyBackgroundNode shader compilation failed.");
				throw std::runtime_error("SkyBackgroundNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
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
			push.sunDirAndAureole = glm::vec4(p.sunDir, 0.0f);
			push.moonDirAndCirrus = glm::vec4(p.moonDir, 0.3f);
			push.sunRadianceAndSkyExp = glm::vec4(p.sunRadiance, p.skyExposure);
			push.worldScale = p.worldScale;
		}

		// SkyViewLUT/TransmittanceLUT are both catalog-converted -- no .Desc() needed. AtmosphereRadiance
		// isn't, so it keeps the exact desc the old hand-rolled Recipe used.
		// Read<TerrainWeatherBiomeTexture>(): humidity coupling for the sun aureole
		// (u_sunAureoleStrength, sky.frag) needs the weather map's moisture channel.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<SkyViewLUT>();
			r.Read<TransmittanceLUT>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Create<AtmosphereRadiance>().Desc(
				graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat)
			);
		}

		void Execute(graph::NodeContext& ctx) {
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();
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

	BRASSICA_REGISTER_NODE(SkyBackgroundNode);

} // namespace brassica
