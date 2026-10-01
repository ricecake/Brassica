#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"
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

	struct TonemapComputeNode: render::NodeRegistrar<TonemapComputeNode> {
		using Resources = graph::Declares<
			graph::Read<HdrColor>,
			graph::Read<GBufferDepth>,
			graph::Modify<AutoExposureBuffer>,
			graph::Create<BloomTextureMip0>,
			graph::Create<BloomTextureMip1>,
			graph::Create<BloomTextureMip2>,
			graph::Create<BloomTextureMip3>,
			graph::Create<BloomTextureMip4>,
			graph::Create<LtmExpTextureMip0>,
			graph::Create<LtmExpTextureMip1>,
			graph::Create<LtmExpTextureMip2>,
			graph::Create<LtmExpTextureMip3>,
			graph::Create<LtmExpTextureMip4>,
			graph::Create<LtmWgtTextureMip0>,
			graph::Create<LtmWgtTextureMip1>,
			graph::Create<LtmWgtTextureMip2>,
			graph::Create<LtmWgtTextureMip3>,
			graph::Create<LtmWgtTextureMip4>,
			graph::Create<LtmFusedTexture>>;

		static constexpr graph::Phase kPhase = brassica::SubPhase::ToneMapping;

		ComputeShader               downsampleShader;
		ComputeShader               upsampleShader;
		ComputeShader               ltmFuseShader;
		render::PipelineLibrary*    pipelineLibrary = nullptr;
		DownsamplePushConstants     downPush{};
		BloomUpsamplePushConstants  upPush{};
		LtmFusePushConstants        fusePush{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			downsampleShader.CompileComputeFromFile(services.device, "shaders/effects/bloom_downsample.comp");
			upsampleShader.CompileComputeFromFile(services.device, "shaders/effects/bloom_upsample.comp");
			ltmFuseShader.CompileComputeFromFile(services.device, "shaders/effects/ltm_fuse.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			s_exposureData.layers[1].targetLuminance = 0.5f;
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

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute, .isActive = s_tonemapComputePassEnabled};
			if (!r.isActive) {
				return r;
			}
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AutoExposureBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::StorageBufferDesc(sizeof(ExposureDataHost)),
				}
			);

			auto addStorageDesc = [&](graph::ResourceId key, std::uint32_t w, std::uint32_t h) {
				r.realizations.push_back(
					graph::ResourceRealization{
						.key = key,
						.access = graph::AccessKind::Write,
						.desc = graph::ComputeStorageImageDesc(w, h, vk::Format::eR16G16B16A16Sfloat),
					}
				);
			};

			std::uint32_t w = ctx.width / 2;
			std::uint32_t h = ctx.height / 2;

			addStorageDesc(graph::IdOf<BloomTextureMip0>(), std::max(1u, w), std::max(1u, h));
			addStorageDesc(graph::IdOf<BloomTextureMip1>(), std::max(1u, w / 2), std::max(1u, h / 2));
			addStorageDesc(graph::IdOf<BloomTextureMip2>(), std::max(1u, w / 4), std::max(1u, h / 4));
			addStorageDesc(graph::IdOf<BloomTextureMip3>(), std::max(1u, w / 8), std::max(1u, h / 8));
			addStorageDesc(graph::IdOf<BloomTextureMip4>(), std::max(1u, w / 16), std::max(1u, h / 16));

			addStorageDesc(graph::IdOf<LtmExpTextureMip0>(), std::max(1u, w), std::max(1u, h));
			addStorageDesc(graph::IdOf<LtmExpTextureMip1>(), std::max(1u, w / 2), std::max(1u, h / 2));
			addStorageDesc(graph::IdOf<LtmExpTextureMip2>(), std::max(1u, w / 4), std::max(1u, h / 4));
			addStorageDesc(graph::IdOf<LtmExpTextureMip3>(), std::max(1u, w / 8), std::max(1u, h / 8));
			addStorageDesc(graph::IdOf<LtmExpTextureMip4>(), std::max(1u, w / 16), std::max(1u, h / 16));

			addStorageDesc(graph::IdOf<LtmWgtTextureMip0>(), std::max(1u, w), std::max(1u, h));
			addStorageDesc(graph::IdOf<LtmWgtTextureMip1>(), std::max(1u, w / 2), std::max(1u, h / 2));
			addStorageDesc(graph::IdOf<LtmWgtTextureMip2>(), std::max(1u, w / 4), std::max(1u, h / 4));
			addStorageDesc(graph::IdOf<LtmWgtTextureMip3>(), std::max(1u, w / 8), std::max(1u, h / 8));
			addStorageDesc(graph::IdOf<LtmWgtTextureMip4>(), std::max(1u, w / 16), std::max(1u, h / 16));

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<LtmFusedTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(std::max(1u, w), std::max(1u, h), vk::Format::eR16G16B16A16Sfloat),
				}
			);

			return r;
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

			std::array<vk::DescriptorSetLayout, 2> downSetLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> downPushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(DownsamplePushConstants)}
			};
			render::ComputePipelineRequest downRequest{
				.shader = &downsampleShader,
				.setLayouts = downSetLayouts,
				.pushConstantRanges = downPushConstantRanges,
			};
			render::ResolvedPipeline downResolved = pipelineLibrary->ResolveCached(downRequest);

			if (downResolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, downResolved.pipeline);
				std::array<vk::DescriptorSet, 2> boundSets{
					static_cast<VkDescriptorSet>(ctx.frameSet),
					static_cast<VkDescriptorSet>(ctx.globalSet)
				};
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, downResolved.layout, 0, boundSets, nullptr);
				vkCmd.pushConstants(downResolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(DownsamplePushConstants), &downPush);

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

			std::array<vk::DescriptorSetLayout, 2> upSetLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> upPushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(BloomUpsamplePushConstants)}
			};
			render::ComputePipelineRequest upRequest{
				.shader = &upsampleShader,
				.setLayouts = upSetLayouts,
				.pushConstantRanges = upPushConstantRanges,
			};
			render::ResolvedPipeline upResolved = pipelineLibrary->ResolveCached(upRequest);

			if (upResolved.pipeline) {
				for (int mip = downPush.numMips - 1; mip > 0; --mip) {
					upPush.srcIndex = bloomMipReadIndex[mip];
					upPush.dstReadIndex = bloomMipReadIndex[mip - 1];
					upPush.dstWriteIndex = bloomMipWriteIndex[mip - 1];
					upPush.srcResolution = bloomMipResolution[mip];
					upPush.dstResolution = bloomMipResolution[mip - 1];

					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, upResolved.pipeline);
					std::array<vk::DescriptorSet, 2> upBoundSets{
						static_cast<VkDescriptorSet>(ctx.frameSet),
						static_cast<VkDescriptorSet>(ctx.globalSet)
					};
					vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, upResolved.layout, 0, upBoundSets, nullptr);
					vkCmd.pushConstants(
						upResolved.layout,
						vk::ShaderStageFlagBits::eCompute,
						0,
						sizeof(BloomUpsamplePushConstants),
						&upPush
					);

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
			}

			// Compute LTM Fuse Pass
			fusePush.expTextureIndex = ctx.Index<LtmExpTextureMip0>();
			fusePush.wgtTextureIndex = ctx.Index<LtmWgtTextureMip0>();
			fusePush.outFusedIndex = ctx.StorageIndex<LtmFusedTexture>();
			fusePush.startMip = 4;
			fusePush.endMip = 0;

			std::array<vk::DescriptorSetLayout, 2> fuseSetLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> fusePushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(LtmFusePushConstants)}
			};
			render::ComputePipelineRequest fuseRequest{
				.shader = &ltmFuseShader,
				.setLayouts = fuseSetLayouts,
				.pushConstantRanges = fusePushConstantRanges,
			};
			render::ResolvedPipeline fuseResolved = pipelineLibrary->ResolveCached(fuseRequest);

			if (fuseResolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, fuseResolved.pipeline);
				std::array<vk::DescriptorSet, 2> boundSets{
					static_cast<VkDescriptorSet>(ctx.frameSet),
					static_cast<VkDescriptorSet>(ctx.globalSet)
				};
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, fuseResolved.layout, 0, boundSets, nullptr);
				vkCmd.pushConstants(fuseResolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(LtmFusePushConstants), &fusePush);

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
		using Resources = graph::Declares<
			graph::Read<HdrColor>,
			graph::Read<GBufferDepth>,
			graph::Read<BloomTextureMip0>,
			graph::Read<LtmFusedTexture>,
			graph::Read<LtmExpTextureMip0>,
			graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = brassica::SubPhase::ToneMapping;

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

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
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

			std::array<GraphicsShader*, 2>         stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>              colorFormats{swapchainFormat};
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};
			std::array<vk::PushConstantRange, 1> pushConstantRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eFragment, 0, sizeof(TonemapPushConstants)}
			};
			render::GraphicsPipelineRequest request{
				.stages = stages,
				.state = kPipelineState,
				.colorFormats = colorFormats,
				.setLayouts = setLayouts,
				.pushConstantRanges = pushConstantRanges,
			};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

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
				sizeof(TonemapPushConstants),
				&push
			);

			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(TonemapComputeNode);
	BRASSICA_REGISTER_NODE(TonemapNode);

} // namespace brassica
