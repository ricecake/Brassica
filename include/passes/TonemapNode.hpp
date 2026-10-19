#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/AutoExposureData.hpp"
#include "types/BloomPushConstants.hpp"
#include "types/CdlGradingData.hpp"
#include "types/TonemapPushConstants.hpp"

namespace brassica {

	struct LtmFusePushConstants {
		std::uint32_t expTextureIndex{0};
		std::uint32_t wgtTextureIndex{0};
		std::uint32_t outFusedIndex{0};
		std::int32_t  startMip{4};
		std::int32_t  endMip{0};
	};

	// S4: no kPhase -- Read<HdrColor>()/Read<GBufferDepth>() are both backed by sole-creator
	// Create-before-accessor edges (DeferredNode, TerrainNode), and Modify<AutoExposureBuffer>()
	// is likewise ordered after Import<AutoExposureBuffer>()'s sole Create (Engine.cpp) --
	// nothing else writes any key this node touches, so there's no writer-pairwise pair left for
	// a phase to disambiguate.
	struct TonemapComputeNode: render::NodeRegistrar<TonemapComputeNode> {
		ComputeShader              downsampleShader;
		ComputeShader              upsampleShader;
		ComputeShader              ltmFuseShader;
		render::PipelineLibrary*   pipelineLibrary = nullptr;
		DownsamplePushConstants    downPush{};
		BloomUpsamplePushConstants upPush{};
		LtmFusePushConstants       fusePush{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			downsampleShader.CompileComputeFromFile(services.device, "shaders/effects/bloom_downsample.comp");
			upsampleShader.CompileComputeFromFile(services.device, "shaders/effects/bloom_upsample.comp");
			ltmFuseShader.CompileComputeFromFile(services.device, "shaders/effects/ltm_fuse.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			s_exposureData.layers[1].targetLuminance = 0.5f;

			if (services.physicalRegistry) {
				graph::ResourceCatalog& catalog = services.physicalRegistry->Catalog();
				catalog.Get<BloomTextureMip0>();
				catalog.Get<BloomTextureMip1>();
				catalog.Get<BloomTextureMip2>();
				catalog.Get<BloomTextureMip3>();
				catalog.Get<BloomTextureMip4>();
				catalog.Get<LtmExpTextureMip0>();
				catalog.Get<LtmExpTextureMip1>();
				catalog.Get<LtmExpTextureMip2>();
				catalog.Get<LtmExpTextureMip3>();
				catalog.Get<LtmExpTextureMip4>();
				catalog.Get<LtmWgtTextureMip0>();
				catalog.Get<LtmWgtTextureMip1>();
				catalog.Get<LtmWgtTextureMip2>();
				catalog.Get<LtmWgtTextureMip3>();
				catalog.Get<LtmWgtTextureMip4>();
				catalog.Get<LtmFusedTexture>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&downsampleShader);
			watcher.RegisterShader(&upsampleShader);
			watcher.RegisterShader(&ltmFuseShader);
		}

		void Destroy(vk::Device device) {
			downsampleShader.Destroy(device);
			upsampleShader.Destroy(device);
			ltmFuseShader.Destroy(device);
		}

		// Active(s_tonemapComputePassEnabled) only gates whether this dispatch runs -- every
		// access below is still declared unconditionally, per the plan's §B rule (TonemapNode.hpp
		// used to return early with no realizations at all when disabled, which would have changed
		// the union graph frame to frame once relations become load-bearing in S3). Dimensions for
		// the 16 mip textures + LtmFusedTexture now come from each key's own catalog Desc()
		// (ResourceKeys.hpp's TonemapMipDesc) instead of this Setup rebuilding them by hand.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Active(s_tonemapComputePassEnabled);
			r.Read<HdrColor>();
			r.Read<GBufferDepth>();
			r.Modify<AutoExposureBuffer>();
			r.Create<BloomTextureMip0>();
			r.Create<BloomTextureMip1>();
			r.Create<BloomTextureMip2>();
			r.Create<BloomTextureMip3>();
			r.Create<BloomTextureMip4>();
			r.Create<LtmExpTextureMip0>();
			r.Create<LtmExpTextureMip1>();
			r.Create<LtmExpTextureMip2>();
			r.Create<LtmExpTextureMip3>();
			r.Create<LtmExpTextureMip4>();
			r.Create<LtmWgtTextureMip0>();
			r.Create<LtmWgtTextureMip1>();
			r.Create<LtmWgtTextureMip2>();
			r.Create<LtmWgtTextureMip3>();
			r.Create<LtmWgtTextureMip4>();
			r.Create<LtmFusedTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// AutoExposureBuffer is imported (Engine::DrawFrame's RegisterImportedBuffer, backed by
			// the real Engine-owned autoExposureBuffers[activeFrame]), not graph-provisioned -- it's
			// already seeded and kept in sync with s_exposureData's tunables every frame from there.
			// No per-node init here: this used to update the frame graph's own transient
			// AutoExposureBuffer allocation, a different buffer than the one this dispatch's
			// ctx.frameSet descriptor actually points at, so the shader never saw it.

			// Compute Downsample Pass
			downPush = s_bloomDownsamplePush;
			downPush.srcResolution = glm::vec2(ctx.width, ctx.height);
			downPush.hdrColorIndex = ctx.Index<HdrColor>();
			downPush.depthIndex = ctx.Index<GBufferDepth>();

			downPush.outMip0Index = ctx.StorageIndex<BloomTextureMip0>();
			downPush.outMip1Index = ctx.StorageIndex<BloomTextureMip1>();
			downPush.outMip2Index = ctx.StorageIndex<BloomTextureMip2>();
			downPush.outMip3Index = ctx.StorageIndex<BloomTextureMip3>();
			downPush.outMip4Index = ctx.StorageIndex<BloomTextureMip4>();

			downPush.outExpMip0Index = ctx.StorageIndex<LtmExpTextureMip0>();
			downPush.outExpMip1Index = ctx.StorageIndex<LtmExpTextureMip1>();
			downPush.outExpMip2Index = ctx.StorageIndex<LtmExpTextureMip2>();
			downPush.outExpMip3Index = ctx.StorageIndex<LtmExpTextureMip3>();
			downPush.outExpMip4Index = ctx.StorageIndex<LtmExpTextureMip4>();

			downPush.outWgtMip0Index = ctx.StorageIndex<LtmWgtTextureMip0>();
			downPush.outWgtMip1Index = ctx.StorageIndex<LtmWgtTextureMip1>();
			downPush.outWgtMip2Index = ctx.StorageIndex<LtmWgtTextureMip2>();
			downPush.outWgtMip3Index = ctx.StorageIndex<LtmWgtTextureMip3>();
			downPush.outWgtMip4Index = ctx.StorageIndex<LtmWgtTextureMip4>();

			render::ResolvedPipeline downResolved =
				render::BindCompute(ctx, *pipelineLibrary, downsampleShader, downPush);

			if (downResolved.pipeline) {
				std::uint32_t groupsX = (ctx.width / 2 + 15) / 16;
				std::uint32_t groupsY = (ctx.height / 2 + 15) / 16;
				vkCmd.dispatch(groupsX, groupsY, 1);

				vk::MemoryBarrier2 barrier(
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
				);
				vk::DependencyInfo dep({}, 1, &barrier);
				vkCmd.pipelineBarrier2(dep);
			}

			// Compute Bloom Upsample Pass -- progressively tent-filters each mip up into the next
			// finer one and additively accumulates it there, coarsest (numMips-1) down to Mip0, so
			// Mip0 ends up holding the fully composited wide-radius bloom TonemapNode reads later.
			// Sequential by construction: each iteration's destination is the next iteration's
			// source, so it can't be collapsed into a single dispatch.
			std::array<std::uint32_t, 5> bloomMipReadIndex{
				ctx.Index<BloomTextureMip0>(),
				ctx.Index<BloomTextureMip1>(),
				ctx.Index<BloomTextureMip2>(),
				ctx.Index<BloomTextureMip3>(),
				ctx.Index<BloomTextureMip4>()
			};
			std::array<std::uint32_t, 5> bloomMipWriteIndex{
				downPush.outMip0Index,
				downPush.outMip1Index,
				downPush.outMip2Index,
				downPush.outMip3Index,
				downPush.outMip4Index
			};
			std::array<glm::vec2, 5> bloomMipResolution{
				glm::vec2(std::max(1u, ctx.width / 2), std::max(1u, ctx.height / 2)),
				glm::vec2(std::max(1u, ctx.width / 4), std::max(1u, ctx.height / 4)),
				glm::vec2(std::max(1u, ctx.width / 8), std::max(1u, ctx.height / 8)),
				glm::vec2(std::max(1u, ctx.width / 16), std::max(1u, ctx.height / 16)),
				glm::vec2(std::max(1u, ctx.width / 32), std::max(1u, ctx.height / 32))
			};

			for (int mip = downPush.numMips - 1; mip > 0; --mip) {
				upPush.srcIndex = bloomMipReadIndex[mip];
				upPush.dstReadIndex = bloomMipReadIndex[mip - 1];
				upPush.dstWriteIndex = bloomMipWriteIndex[mip - 1];
				upPush.srcResolution = bloomMipResolution[mip];
				upPush.dstResolution = bloomMipResolution[mip - 1];

				render::ResolvedPipeline upResolved =
					render::BindCompute(ctx, *pipelineLibrary, upsampleShader, upPush);
				if (!upResolved.pipeline) {
					continue;
				}

				std::uint32_t dstW = static_cast<std::uint32_t>(bloomMipResolution[mip - 1].x);
				std::uint32_t dstH = static_cast<std::uint32_t>(bloomMipResolution[mip - 1].y);
				vkCmd.dispatch((dstW + 15) / 16, (dstH + 15) / 16, 1);

				vk::MemoryBarrier2 upBarrier(
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
				);
				vk::DependencyInfo upDep({}, 1, &upBarrier);
				vkCmd.pipelineBarrier2(upDep);
			}

			// Compute LTM Fuse Pass
			fusePush.expTextureIndex = ctx.Index<LtmExpTextureMip0>();
			fusePush.wgtTextureIndex = ctx.Index<LtmWgtTextureMip0>();
			fusePush.outFusedIndex = ctx.StorageIndex<LtmFusedTexture>();
			fusePush.startMip = 4;
			fusePush.endMip = 0;

			render::ResolvedPipeline fuseResolved = render::BindCompute(ctx, *pipelineLibrary, ltmFuseShader, fusePush);

			if (fuseResolved.pipeline) {
				std::uint32_t groupsX = (ctx.width / 2 + 15) / 16;
				std::uint32_t groupsY = (ctx.height / 2 + 15) / 16;
				vkCmd.dispatch(groupsX, groupsY, 1);

				vk::MemoryBarrier2 barrier(
					vk::PipelineStageFlagBits2::eComputeShader,
					vk::AccessFlagBits2::eShaderStorageWrite,
					vk::PipelineStageFlagBits2::eFragmentShader,
					vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead
				);
				vk::DependencyInfo dep({}, 1, &barrier);
				vkCmd.pipelineBarrier2(dep);
			}
		}
	};

	struct TonemapNode: render::NodeRegistrar<TonemapNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		VertexShader   vertShader;
		FragmentShader fragShader;
		vk::Format     swapchainFormat = vk::Format::eUndefined;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		TonemapPushConstants     push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			swapchainFormat = services.swapchainFormat;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/tonemap.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/tonemap.frag")) {
				spdlog::critical("TonemapNode shader compilation failed.");
				throw std::runtime_error("TonemapNode shader compilation failed.");
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

		void SetFrameParams(const render::NodeFrameParams& /*p*/) { push = s_tonemapPush; }

		// No .Desc() on Swapchain -- it's Imported, so Provision() skips its realization's desc
		// entirely regardless of content (same reasoning as ImGuiNode.hpp). The three mip reads
		// are safe to declare now that TonemapComputeNode (the real producer) creates them
		// through the catalog instead of a hand-rolled Recipe -- Create-before-accessor orders
		// it before this node regardless of either side's phase. S4: no kPhase -- this node is
		// the ToneMap phase's designated keystone (design plan's §D table): ImGuiNode's
		// After<TonemapNode>() (ImGuiNode.hpp) is the only other thing that currently needs "after
		// ToneMap" as a named anchor, and Keystone<>() is what makes that guarantee hold even if
		// a future ToneMap-phase node runs unordered relative to this one.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Keystone<brassica::phase::ToneMap>();
			r.Read<HdrColor>();
			r.Read<GBufferDepth>();
			r.Read<BloomTextureMip0>();
			r.Read<LtmFusedTexture>();
			r.Read<LtmExpTextureMip0>();
			r.Modify<Swapchain>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// Tone Mapping Compositing Graphics Pass
			push = s_tonemapPush;
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.bloomBlurIndex = ctx.Index<BloomTextureMip0>();
			push.ltmFusedIndex = ctx.Index<LtmFusedTexture>();
			push.ltmExpMipIndex = ctx.Index<LtmExpTextureMip0>();
			push.depthTextureIndex = ctx.Index<GBufferDepth>();
			push.ltmRes = glm::vec2(ctx.width / 2, ctx.height / 2);
			push.numCdlEntries = s_cdlGradingLayers.numEntries;

			std::array<GraphicsShader*, 2> stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{swapchainFormat};
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

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(TonemapComputeNode);
	BRASSICA_REGISTER_NODE(TonemapNode);

} // namespace brassica
