#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "constants.h"
#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "lighting/ILightManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceGroups.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	class ShaderWatcher;

	struct DeferredPushConstants {
		glm::uvec4 gridParams{
			constants::Class::Terrain::DefaultMaxLODs,
			constants::Class::Terrain::MeshletsPerRow,
			constants::Class::Terrain::TotalMeshlets,
			constants::Class::Terrain::MapDim
		}; // x = numLODs, y = meshletsPerRow, z = totalMeshlets, w = textureDim
		std::uint32_t gPositionIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t backgroundIndex{0};
		std::uint32_t clipmapIndex{0};
		std::uint32_t tlasIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t minMaxIndex{0};
		std::uint32_t biomeIndex{0};
		std::uint32_t weatherBiomeIndex{0};
		std::uint32_t gMaterialIndex{0};
		std::uint32_t skyViewIndex{0};
		std::uint32_t gtaoIndex{0};
		float         aoDirectInfluence{0.0f};
		std::uint32_t debugVisualizeGTAO{0};
		std::uint32_t cloudShadowIndex{0};
	};

	struct DeferredNode: render::NodeRegistrar<DeferredNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		vk::Format               swapchainFormat = vk::Format::eUndefined;
		DeferredPushConstants    push{};
		bool                     enableGTAO{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			swapchainFormat = services.swapchainFormat;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/deferred.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/deferred.frag")) {
				spdlog::critical("DeferredNode shader compilation failed.");
				throw std::runtime_error("DeferredNode shader compilation failed.");
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
			push.gridParams = p.terrainGridParams;
			if (ServiceLocator::Instance().Has<ILightManager>()) {
				auto lightManager = ServiceLocator::Instance().Get<ILightManager>();
				enableGTAO = lightManager->IsGTAOEnabled();
				push.aoDirectInfluence = enableGTAO ? lightManager->GetGTAODirectInfluence() : 0.0f;
				push.debugVisualizeGTAO = (enableGTAO && lightManager->IsGTAODebugVisualizeEnabled()) ? 1u : 0u;
			}
		}

		// Reads every GBuffer key (genuinely, unlike AtmosphereCompositeNode/CirrusNode/
		// CloudRenderNode/VolumetricRenderNode below -- this is the one real exception lesson
		// #2's writeup calls out). No Read<AtmosphereRadiance>() or Read<TerrainTLAS>(): neither
		// is catalog-converted, and AtmosphereRadiance's producer (SkyBackgroundNode) schedules
		// after DeferredShading anyway (there's no real edge to restore). This node's own
		// Resources never declared TerrainTLAS as anything but a Read either, so dropping it
		// costs nothing today; see TerrainGenNode.hpp's SetupBuilder::AccessHandle::Desc comment
		// for why an AccelerationStructure-kind key needs that escape hatch on *every* toucher,
		// not just its producer, if it's ever read through the builder instead of omitted.
		// Read<TerrainWeatherBiomeTexture>() restored (S4): catalog-converted since batch 5, so
		// the lesson #10 hazard that justified omitting it is gone -- this was simply never
		// re-added when that batch landed, found and fixed empirically while migrating this
		// node's phase.
		// S4: DeferredNode is the Lighting phase's designated keystone (design plan's §D table) --
		// everything it reads (GTAO/ClusterLight/CloudShadow/SkyView/terrain/GBuffer) already
		// gives it a real Create-before-accessor edge from each producer, and it's the sole
		// Create<> of HdrColor, which already orders it before every HdrColor accessor
		// downstream (Composite phase and beyond) -- so Keystone<Lighting>() changes nothing
		// about *this* node's own schedule position today. What it does is give any future node
		// that needs "after the whole Lighting phase, whatever that ends up including" a single,
		// real name to say After<DeferredNode>() against, instead of enumerating producers.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Keystone<brassica::phase::Lighting>();
			r.Read<GBufferPosition>();
			r.Read<GBufferNormal>();
			r.Read<GBufferAlbedo>();
			r.Read<GBufferMaterial>();
			r.Read<GBufferDepth>();
			r.Read<ClusteredLighting>();
			r.Read<TerrainClipmapTexture>();
			r.Read<TerrainMinMaxTexture>();
			r.Read<TerrainBiomeTexture>();
			r.Read<TerrainWeatherBiomeTexture>();
			r.Read<SkyViewLUT>();
			r.Read<GTAOTexture>();
			r.Read<CloudShadowMap>();
			r.Create<HdrColor>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.gMaterialIndex = ctx.Index<GBufferMaterial>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.backgroundIndex = ctx.Index<AtmosphereRadiance>();
			push.clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			push.tlasIndex = ctx.Index<TerrainTLAS>();
			push.minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			push.biomeIndex = ctx.Index<TerrainBiomeTexture>();
			push.weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.gtaoIndex = enableGTAO ? ctx.Index<GTAOTexture>() : 0u;
			push.cloudShadowIndex = ctx.Index<CloudShadowMap>();

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{vk::Format::eR16G16B16A16Sfloat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
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

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, resolved.layout, 0, boundSets, nullptr);
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

	BRASSICA_REGISTER_NODE(DeferredNode);

} // namespace brassica
