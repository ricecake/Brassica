#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/AutoExposureData.hpp"
#include "types/TonemapPushConstants.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	struct DownsamplePushConstants {
		glm::vec2     srcResolution{0.0f, 0.0f};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t depthIndex{0};
		std::uint32_t outMip0Index{0};
		std::uint32_t outMip1Index{0};
		std::uint32_t outMip2Index{0};
		std::uint32_t outMip3Index{0};
		std::uint32_t outMip4Index{0};
		std::uint32_t outExpMip0Index{0};
		std::uint32_t outExpMip1Index{0};
		std::uint32_t outExpMip2Index{0};
		std::uint32_t outExpMip3Index{0};
		std::uint32_t outExpMip4Index{0};
		std::uint32_t outWgtMip0Index{0};
		std::uint32_t outWgtMip1Index{0};
		std::uint32_t outWgtMip2Index{0};
		std::uint32_t outWgtMip3Index{0};
		std::uint32_t outWgtMip4Index{0};
		std::int32_t  numMips{5};
		float         threshold{1.0f};
		float         deltaTime{0.016f};
	};

	struct AutoExposureUpdatePushConstants {
		float deltaTime{0.016f};
	};

	struct LtmFusePushConstants {
		std::uint32_t expTextureIndex{0};
		std::uint32_t wgtTextureIndex{0};
		std::uint32_t outFusedIndex{0};
		std::int32_t  startMip{4};
		std::int32_t  endMip{0};
	};

	struct TonemapNode: render::NodeRegistrar<TonemapNode> {
		using Resources = graph::Declares<
			graph::Read<HdrColor>,
			graph::Read<GBufferDepth>,
			graph::Create<AutoExposureBuffer>,
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
			graph::Create<LtmFusedTexture>,
			graph::Modify<Swapchain>>;

		static constexpr graph::Phase kPhase = brassica::SubPhase::ToneMapping;

		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            downsampleShader;
		ComputeShader            aeUpdateShader;
		ComputeShader            ltmFuseShader;
		VertexShader             vertShader;
		FragmentShader           fragShader;
		vk::Format               swapchainFormat = vk::Format::eUndefined;
		bool                     initializedBufferParams = false;
		TonemapPushConstants     push{};
		DownsamplePushConstants  downPush{};
		AutoExposureUpdatePushConstants aeUpdatePush{};
		LtmFusePushConstants     fusePush{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			swapchainFormat = services.swapchainFormat;
			downsampleShader.CompileComputeFromFile(services.device, "shaders/effects/bloom_downsample.comp");
			aeUpdateShader.CompileComputeFromFile(services.device, "shaders/effects/autoexposure_update.comp");
			ltmFuseShader.CompileComputeFromFile(services.device, "shaders/effects/ltm_fuse.comp");
			vertShader.CompileVertexFromFile(services.device, "shaders/tonemap.vert");
			fragShader.CompileFragmentFromFile(services.device, "shaders/tonemap.frag");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			s_exposureData.layers[1].targetLuminance = 0.5f;
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&downsampleShader);
			watcher.RegisterShader(&aeUpdateShader);
			watcher.RegisterShader(&ltmFuseShader);
			watcher.RegisterShader(&vertShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			downsampleShader.Destroy(device);
			aeUpdateShader.Destroy(device);
			ltmFuseShader.Destroy(device);
			vertShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& /*p*/) { push = s_tonemapPush; }

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Graphics};
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<HdrColor>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<GBufferDepth>(),
					.access = graph::AccessKind::Read,
					.desc = graph::DepthAttachmentDesc(ctx.width, ctx.height, vk::Format::eD32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AutoExposureBuffer>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::BufferDesc{
						.sizeBytes = sizeof(ExposureDataHost),
						.usageMask = static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst),
					},
				}
			);

			auto addStorageDesc = [&](graph::ResourceKeyId key, std::uint32_t w, std::uint32_t h) {
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

			auto aeBuffer = ctx.GetBuffer<AutoExposureBuffer>();
			if (aeBuffer && aeBuffer->GetBuffer() && !initializedBufferParams) {
				vkCmd.updateBuffer(
					aeBuffer->GetBuffer(),
					0,
					sizeof(ExposureDataHost),
					&s_exposureData
				);
				initializedBufferParams = true;

				vk::MemoryBarrier2 barrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
					.srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
				};
				vk::DependencyInfo dep{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
				vkCmd.pipelineBarrier2(dep);
			}

			float dt = 0.016f;

			// Stage 1: Compute Downsample & Histogram Accumulation
			downPush.srcResolution = glm::vec2(ctx.width, ctx.height);
			downPush.hdrColorIndex = ctx.Index<HdrColor>();
			downPush.depthIndex = ctx.Index<GBufferDepth>();
			downPush.deltaTime = dt;

			downPush.outMip0Index = ctx.Index<BloomTextureMip0>();
			downPush.outMip1Index = ctx.Index<BloomTextureMip1>();
			downPush.outMip2Index = ctx.Index<BloomTextureMip2>();
			downPush.outMip3Index = ctx.Index<BloomTextureMip3>();
			downPush.outMip4Index = ctx.Index<BloomTextureMip4>();

			downPush.outExpMip0Index = ctx.Index<LtmExpTextureMip0>();
			downPush.outExpMip1Index = ctx.Index<LtmExpTextureMip1>();
			downPush.outExpMip2Index = ctx.Index<LtmExpTextureMip2>();
			downPush.outExpMip3Index = ctx.Index<LtmExpTextureMip3>();
			downPush.outExpMip4Index = ctx.Index<LtmExpTextureMip4>();

			downPush.outWgtMip0Index = ctx.Index<LtmWgtTextureMip0>();
			downPush.outWgtMip1Index = ctx.Index<LtmWgtTextureMip1>();
			downPush.outWgtMip2Index = ctx.Index<LtmWgtTextureMip2>();
			downPush.outWgtMip3Index = ctx.Index<LtmWgtTextureMip3>();
			downPush.outWgtMip4Index = ctx.Index<LtmWgtTextureMip4>();

			render::ComputePipelineRequest downRequest{
				.computeStage = &downsampleShader,
				.setLayouts = std::array<vk::DescriptorSetLayout, 2>{
					static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
					static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
				},
				.pushConstantRanges = std::array<vk::PushConstantRange, 1>{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(DownsamplePushConstants)}
				},
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

				vk::MemoryBarrier2 barrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
				};
				vk::DependencyInfo dep{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
				vkCmd.pipelineBarrier2(dep);
			}

			// Stage 2: 1x1 Workgroup Histogram Reduction and Eye Adaptation Update
			aeUpdatePush.deltaTime = dt;
			render::ComputePipelineRequest aeUpdateRequest{
				.computeStage = &aeUpdateShader,
				.setLayouts = std::array<vk::DescriptorSetLayout, 2>{
					static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
					static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
				},
				.pushConstantRanges = std::array<vk::PushConstantRange, 1>{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(AutoExposureUpdatePushConstants)}
				},
			};
			render::ResolvedPipeline aeUpdateResolved = pipelineLibrary->ResolveCached(aeUpdateRequest);

			if (aeUpdateResolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, aeUpdateResolved.pipeline);
				std::array<vk::DescriptorSet, 2> boundSets{
					static_cast<VkDescriptorSet>(ctx.frameSet),
					static_cast<VkDescriptorSet>(ctx.globalSet)
				};
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, aeUpdateResolved.layout, 0, boundSets, nullptr);
				vkCmd.pushConstants(aeUpdateResolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(AutoExposureUpdatePushConstants), &aeUpdatePush);

				vkCmd.dispatch(1, 1, 1);

				vk::MemoryBarrier2 barrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead,
				};
				vk::DependencyInfo dep{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
				vkCmd.pipelineBarrier2(dep);
			}

			// Stage 3: Compute LTM Fuse Pass
			fusePush.expTextureIndex = ctx.Index<LtmExpTextureMip0>();
			fusePush.wgtTextureIndex = ctx.Index<LtmWgtTextureMip0>();
			fusePush.outFusedIndex = ctx.Index<LtmFusedTexture>();
			fusePush.startMip = 4;
			fusePush.endMip = 0;

			render::ComputePipelineRequest fuseRequest{
				.computeStage = &ltmFuseShader,
				.setLayouts = std::array<vk::DescriptorSetLayout, 2>{
					static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
					static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
				},
				.pushConstantRanges = std::array<vk::PushConstantRange, 1>{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(LtmFusePushConstants)}
				},
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

				vk::MemoryBarrier2 barrier{
					.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
					.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
					.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
					.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead,
				};
				vk::DependencyInfo dep{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
				vkCmd.pipelineBarrier2(dep);
			}

			// Stage 4: Tone Mapping Compositing Graphics Pass
			push = s_tonemapPush;
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.bloomBlurIndex = ctx.Index<BloomTextureMip0>();
			push.ltmFusedIndex = ctx.Index<LtmFusedTexture>();
			push.ltmExpMipIndex = ctx.Index<LtmExpTextureMip0>();
			push.depthTextureIndex = ctx.Index<GBufferDepth>();
			push.ltmRes = glm::vec2(ctx.width / 2, ctx.height / 2);

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

	BRASSICA_REGISTER_NODE(TonemapNode);

} // namespace brassica
