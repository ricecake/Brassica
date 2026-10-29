#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct alignas(16) CloudRaymarchPushConstants {
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 sunRadianceAndSkyExp{3.0f, 2.94f, 2.76f, 1.0f};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t transmittanceLUTIndex{0};
		std::uint32_t skyViewIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t cloudShadowIndex{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t volumeIndex{0}; // sampled bindless index for CloudDetailVolume
		std::uint32_t colorStorageIdx{0};
		std::uint32_t transmittanceStorageIdx{0};
		std::uint32_t depthStorageIdx{0};
		std::uint32_t velocityStorageIdx{0};
		std::uint32_t outputWidth{0};
		std::uint32_t outputHeight{0};
		float         deltaTime{0.016f};
	};

	// Raymarches CloudDetailVolume into CloudColor/CloudTransmittance (the visible result) and
	// CloudDepth/CloudVelocity (reprojection data for the tile-based temporal-accumulation
	// plan's Stage B). Now a compute pass, not a graphics/fragment pass -- a prerequisite for
	// Stage D's indirect per-tile dispatch -- so it no longer touches HdrColor directly;
	// CloudCompositeNode does that blend instead, and has taken over this node's old position in
	// the HdrColor writer chain (After<CirrusNode>() etc.) since it's the one actually Modifying
	// that resource now.
	struct CloudRenderNode: render::NodeRegistrar<CloudRenderNode> {
		render::PipelineLibrary*     pipelineLibrary = nullptr;
		ComputeShader                raymarchShader; // shaders/cloud_raymarch.comp
		CloudRaymarchPushConstants   push{};
		std::uint32_t                width{0};
		std::uint32_t                height{0};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!raymarchShader.CompileComputeFromFile(services.device, "shaders/cloud_raymarch.comp")) {
				spdlog::critical("CloudRenderNode shader compilation failed.");
				throw std::runtime_error("CloudRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&raymarchShader);
			}
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<CloudColor>();
				catalog.Get<CloudTransmittance>();
				catalog.Get<CloudDepth>();
				catalog.Get<CloudVelocity>();
			}
		}

		void Destroy(vk::Device device) { raymarchShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.sunDir = glm::vec4(p.sunDir, 0.0f);
			push.sunRadianceAndSkyExp = glm::vec4(p.sunRadiance, p.skyExposure);
		}

		// No After<CirrusNode>() here anymore -- that ordering existed only because this node
		// used to Modify<HdrColor>() alongside Cirrus/Water/Particle/Lightning. CloudCompositeNode
		// inherits that position now; this node's only real dependencies are its own reads below.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			r.Domain(graph::ExecutionDomain::Compute);
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<CloudDetailVolume>();
			r.Read<TransmittanceLUT>();
			r.Read<SkyViewLUT>();
			r.Read<CloudShadowMap>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Read<TerrainClipmapTexture>();
			r.Create<CloudColor>();
			r.Create<CloudTransmittance>();
			r.Create<CloudDepth>();
			r.Create<CloudVelocity>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.volumeIndex = ctx.Index<CloudDetailVolume>();
			push.transmittanceLUTIndex = ctx.Index<TransmittanceLUT>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.cloudShadowIndex = ctx.Index<CloudShadowMap>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.colorStorageIdx = ctx.StorageIndex<CloudColor>();
			push.transmittanceStorageIdx = ctx.StorageIndex<CloudTransmittance>();
			push.depthStorageIdx = ctx.StorageIndex<CloudDepth>();
			push.velocityStorageIdx = ctx.StorageIndex<CloudVelocity>();
			push.outputWidth = width;
			push.outputHeight = height;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, raymarchShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.dispatch((width + 7) / 8, (height + 7) / 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(CloudRenderNode);

} // namespace brassica
