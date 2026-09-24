#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <glm/glm.hpp>
#include "vulkan/vulkan.hpp"

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
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
					.desc = graph::DepthBufferDesc(ctx.width, ctx.height),
				}
			);

			graph::ResourceDesc aeDesc = graph::MappedStorageBufferDesc(sizeof(ExposureDataHost));
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<AutoExposureBuffer>(),
					.access = graph::AccessKind::Write,
					.desc = aeDesc,
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

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Swapchain>(),
					.access = graph::AccessKind::ReadWrite,
					.desc = graph::ColorAttachmentDesc(ctx.width, ctx.height, swapchainFormat),
				}
			);
			return r;
		}

		void ReadBackAutoExposureStats(graph::NodeContext& ctx) {
			if (ctx.resources) {
				if (const auto* registry = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
					if (auto physBuf = registry->GetBuffer<AutoExposureBuffer>()) {
						if (physBuf->IsHostMapped()) {
							if (const void* mapped = physBuf->MappedSlice(ctx.frameIndex)) {
								const auto* gpuData = static_cast<const ExposureDataHost*>(mapped);
								for (int i = 0; i < 2; i++) {
									s_exposureData.layers[i].adaptedLuminance = gpuData->layers[i].adaptedLuminance;
									s_exposureData.layers[i].minLuma = gpuData->layers[i].minLuma;
									s_exposureData.layers[i].maxLuma = gpuData->layers[i].maxLuma;
									s_exposureData.layers[i].avgLuma = gpuData->layers[i].avgLuma;
									s_exposureData.layers[i].stdDevLuma = gpuData->layers[i].stdDevLuma;
									s_exposureData.layers[i].emaMinLuma = gpuData->layers[i].emaMinLuma;
									s_exposureData.layers[i].emaMaxLuma = gpuData->layers[i].emaMaxLuma;
									s_exposureData.layers[i].emaAvgLuma = gpuData->layers[i].emaAvgLuma;
									s_exposureData.layers[i].emaStdDevLuma = gpuData->layers[i].emaStdDevLuma;
									s_exposureData.layers[i].autoUchimuraP = gpuData->layers[i].autoUchimuraP;
									s_exposureData.layers[i].autoUchimuraA = gpuData->layers[i].autoUchimuraA;
									s_exposureData.layers[i].autoUchimuraM = gpuData->layers[i].autoUchimuraM;
									s_exposureData.layers[i].autoUchimuraL = gpuData->layers[i].autoUchimuraL;
									s_exposureData.layers[i].autoUchimuraC = gpuData->layers[i].autoUchimuraC;
									s_exposureData.layers[i].autoUchimuraB = gpuData->layers[i].autoUchimuraB;
									for (int b = 0; b < 256; b++) {
										s_exposureData.layers[i].histogram[b] = gpuData->layers[i].histogram[b];
									}
								}
							}
						}
					}
				}
			}
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// Upload Host Exposure parameters
			ctx.WriteSpan<AutoExposureBuffer>(std::span<const ExposureDataHost>(&s_exposureData, 1));

			float dt = 0.016f;

			// Stage 1: Compute Downsample & Histogram Accumulation
			downPush.srcResolution = glm::vec2(ctx.width, ctx.height);
			downPush.hdrColorIndex = ctx.Index<HdrColor>();
			downPush.depthIndex = ctx.Index<GBufferDepth>();
			downPush.deltaTime = dt;

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

			render::ComputePipelineRequest downRequest{
				.shader = &downsampleShader,
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

				vk::MemoryBarrier memoryBarrier(
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead
				);
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader,
					vk::DependencyFlags{},
					1, &memoryBarrier,
					0, nullptr,
					0, nullptr
				);
			}

			// Stage 2: 1x1 Workgroup Histogram Reduction and Eye Adaptation Update
			aeUpdatePush.deltaTime = dt;
			render::ComputePipelineRequest aeUpdateRequest{
				.shader = &aeUpdateShader,
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

				vk::MemoryBarrier memoryBarrier(
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead
				);
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eFragmentShader,
					vk::DependencyFlags{},
					1, &memoryBarrier,
					0, nullptr,
					0, nullptr
				);
			}

			// Stage 3: Compute LTM Fuse Pass
			fusePush.expTextureIndex = ctx.Index<LtmExpTextureMip0>();
			fusePush.wgtTextureIndex = ctx.Index<LtmWgtTextureMip0>();
			fusePush.outFusedIndex = ctx.StorageIndex<LtmFusedTexture>();
			fusePush.startMip = 4;
			fusePush.endMip = 0;

			render::ComputePipelineRequest fuseRequest{
				.shader = &ltmFuseShader,
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

				vk::MemoryBarrier memoryBarrier(
					vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead
				);
				vkCmd.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eFragmentShader,
					vk::DependencyFlags{},
					1, &memoryBarrier,
					0, nullptr,
					0, nullptr
				);
			}

			// Read back updated exposure statistics for live UI histogram & metrics
			ReadBackAutoExposureStats(ctx);

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
