#pragma once

#include <array>
#include <cstdint>

#include "VulkanCompat.hpp"

#ifndef BRASSICA_HAS_VULKAN
	#if __has_include(<vulkan/vulkan.hpp>) || __has_include("vulkan/vulkan.hpp")
		#define BRASSICA_HAS_VULKAN 1
	#else
		#define BRASSICA_HAS_VULKAN 0
	#endif
#endif

#if BRASSICA_HAS_VULKAN
	#include "cloud/ICloudManager.hpp"
	#include "graph/PhysicalResource.hpp"
	#include "render/PipelineLibrary.hpp"
	#include "ServiceLocator.hpp"
	#include "Shader.hpp"
	#include "ShaderWatcher.hpp"
	#include "types/CloudPushConstants.hpp"
#else
namespace brassica {
	class ShaderWatcher;

	namespace render {
		class PipelineLibrary;
	} // namespace render
} // namespace brassica
#endif

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"

namespace brassica {

	// Boidish Cloud System Nodes
	struct CloudBakeNode : render::NodeRegistrar<CloudBakeNode> {
		using Resources = graph::Declares<
			graph::Create<CloudWeatherTexture>,
			graph::Create<CloudWeatherMinMaxTexture>,
			graph::Create<Cloud3DVolumeTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::Prepare;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader weatherShader;
		ComputeShader volumeShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			weatherShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_weather_bake.comp");
			volumeShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_3d_volume_bake.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&weatherShader);
			watcher.RegisterShader(&volumeShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			weatherShader.Destroy(device);
			volumeShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudWeatherTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(1024, 1024, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudWeatherMinMaxTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(1024, 1024, vk::Format::eR16G16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<Cloud3DVolumeTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(128, 128, 128, vk::Format::eR16G16B16A16Sfloat),
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudWeatherTexture>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = 1024,
					.height = 1024,
					.formatCode = 97, // eR16G16B16A16Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudWeatherMinMaxTexture>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = 1024,
					.height = 1024,
					.formatCode = 83, // eR16G16Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<Cloud3DVolumeTexture>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image3D,
					.width = 128,
					.height = 128,
					.depth = 128,
					.formatCode = 97, // eR16G16B16A16Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			CloudBakePushConstants push{};
			push.weatherStorageIdx = ctx.StorageIndex<CloudWeatherTexture>();
			push.minMaxStorageIdx = ctx.StorageIndex<CloudWeatherMinMaxTexture>();
			push.volumeStorageIdx = ctx.StorageIndex<Cloud3DVolumeTexture>();
			push.worldScale = frameParams.worldScale;
			push.cloudCoverage = cloudState.coverage;
			push.cloudThickness = cloudState.thickness;
			push.time = frameParams.time;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};

			render::ComputePipelineRequest reqW{.shader = &weatherShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resW = pipelineLibrary->ResolveCached(reqW);
			if (resW.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resW.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resW.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(resW.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudBakePushConstants), &push);
			vkCmd.dispatch(64, 64, 1);

			render::ComputePipelineRequest reqV{.shader = &volumeShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resV = pipelineLibrary->ResolveCached(reqV);
			if (resV.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resV.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resV.layout, 0, boundSets, nullptr);
			}
			vkCmd.pushConstants(resV.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudBakePushConstants), &push);
			vkCmd.dispatch(32, 32, 32);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudBoundingNode : render::NodeRegistrar<CloudBoundingNode> {
		using Resources = graph::Declares<
			graph::Read<GBufferDepth>,
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Create<CloudBoundingTexture>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_bounding.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudBoundingTexture>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudBoundingTexture>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 109, // eR32G32B32A32Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			CloudBoundingPushConstants push{};
			push.depthTextureIdx = ctx.Index<GBufferDepth>();
			push.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>();
			push.boundingStorageIdx = ctx.StorageIndex<CloudBoundingTexture>();
			push.cloudMaxRayDistance = cloudState.maxRayDistance;
			push.renderScale = cloudState.renderScale;
			push.worldScale = frameParams.worldScale;
			push.cloudCoverage = cloudState.coverage;
			push.cloudAltitude = cloudState.altitude;
			push.cloudThickness = cloudState.thickness;
			push.time = frameParams.time;
			push.frameIndex = static_cast<uint32_t>(ctx.frameIndex);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudBoundingPushConstants), &push);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudShadowBakeNode : render::NodeRegistrar<CloudShadowBakeNode> {
		using Resources = graph::Declares<
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Create<CloudShadowMap>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_shadow_bake.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& /*ctx*/) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			auto shadowDesc = graph::ComputeStorageImageDesc(512, 512, vk::Format::eR16Sfloat);
			shadowDesc.layers = 8;
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudShadowMap>(),
					.access = graph::AccessKind::Write,
					.desc = shadowDesc,
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudShadowMap>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = 512,
					.height = 512,
					.layers = 8,
					.formatCode = 76, // eR16Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			CloudShadowBakePushConstants push{};
			push.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>();
			push.shadowMapStorageIdx = ctx.StorageIndex<CloudShadowMap>();
			push.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
			push.worldScale = frameParams.worldScale;
			push.cloudAltitude = cloudState.altitude;
			push.cloudThickness = cloudState.thickness;
			push.primaryLightDir = frameParams.sunDir;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudShadowBakePushConstants), &push);

			vkCmd.dispatch(64, 64, 1);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudTileSchedulerNode : render::NodeRegistrar<CloudTileSchedulerNode> {
		using Resources = graph::Declares<
			graph::Read<CloudBoundingTexture>,
			graph::Create<CloudErrorMap>,
			graph::Create<CloudTileQueueSSBO>,
			graph::Create<CloudIndirectDispatchSSBO>>;

		static constexpr graph::Phase kPhase = SubPhase::GIComput;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_tile_scheduler.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;

#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudErrorMap>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(tileCols, tileRows, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTileQueueSSBO>(),
					.access = graph::AccessKind::Write,
					.desc = graph::StorageBufferDesc((tileCols * tileRows + 16) * sizeof(uint32_t) * 2),
				}
			);
			graph::ResourceDesc indirectDesc = graph::StorageBufferDesc(3 * sizeof(uint32_t));
			indirectDesc.usageMask |= static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eIndirectBuffer);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudIndirectDispatchSSBO>(),
					.access = graph::AccessKind::Write,
					.desc = indirectDesc,
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudErrorMap>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = tileCols,
					.height = tileRows,
					.formatCode = 109, // eR32G32B32A32Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudTileQueueSSBO>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Buffer,
					.byteSize = (tileCols * tileRows + 16) * sizeof(uint32_t) * 2,
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudIndirectDispatchSSBO>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Buffer,
					.byteSize = 3 * sizeof(uint32_t),
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			auto* registry = static_cast<graph::PhysicalResourceRegistry*>(ctx.resources);

			CloudTileSchedulerPushConstants push{};
			push.boundingMapIdx = ctx.Index<CloudBoundingTexture>();
			push.errorMapStorageIdx = ctx.StorageIndex<CloudErrorMap>();
			push.tileQueueBufferAddr = registry ? registry->GetBufferDeviceAddress<CloudTileQueueSSBO>() : 0;
			push.indirectDispatchBufferAddr = registry ? registry->GetBufferDeviceAddress<CloudIndirectDispatchSSBO>() : 0;
			push.pass = 0;
			push.priorityErrorWeight = cloudState.priorityErrorWeight;
			push.priorityGradWeight = cloudState.priorityGradWeight;
			push.priorityAgeWeight = cloudState.priorityAgeWeight;
			push.priorityNeighborErrorWeight = cloudState.priorityNeighborErrorWeight;
			push.priorityNeighborGradWeight = cloudState.priorityNeighborGradWeight;
			push.priorityThreshold = cloudState.priorityThreshold;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTileSchedulerPushConstants), &push);

			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;
			uint32_t gx = (tileCols + 7) / 8;
			uint32_t gy = (tileRows + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudRenderNode : render::NodeRegistrar<CloudRenderNode> {
		using Resources = graph::Declares<
			graph::Read<GBufferDepth>,
			graph::Read<CloudBoundingTexture>,
			graph::Read<CloudWeatherMinMaxTexture>,
			graph::Read<CloudWeatherTexture>,
			graph::Read<Cloud3DVolumeTexture>,
			graph::Read<CloudTileQueueSSBO>,
			graph::Read<CloudIndirectDispatchSSBO>,
			graph::Create<CloudPackedColor>,
			graph::Create<CloudPackedDepth>,
			graph::Create<CloudPackedVelocity>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_render.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedDepth>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudPackedVelocity>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16Sfloat),
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudPackedColor>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 97, // eR16G16B16A16Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudPackedDepth>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 109, // eR32G32B32A32Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudPackedVelocity>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 83, // eR16G16Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			auto* registry = static_cast<graph::PhysicalResourceRegistry*>(ctx.resources);

			CloudRenderPushConstants push{};
			push.depthTextureIdx = ctx.Index<GBufferDepth>();
			push.boundingMapIdx = ctx.Index<CloudBoundingTexture>();
			push.weatherMinMaxIdx = ctx.Index<CloudWeatherMinMaxTexture>();
			push.weatherTextureIdx = ctx.Index<CloudWeatherTexture>();
			push.volume3DIdx = ctx.Index<Cloud3DVolumeTexture>();
			push.packedColorStorageIdx = ctx.StorageIndex<CloudPackedColor>();
			push.packedDepthStorageIdx = ctx.StorageIndex<CloudPackedDepth>();
			push.packedVelocityStorageIdx = ctx.StorageIndex<CloudPackedVelocity>();
			push.errorMapStorageIdx = ctx.StorageIndex<CloudErrorMap>();
			push.tileQueueBufferAddr = registry ? registry->GetBufferDeviceAddress<CloudTileQueueSSBO>() : 0;
			push.cloudMaxRayDistance = cloudState.maxRayDistance;
			push.renderScale = cloudState.renderScale;
			push.worldScale = frameParams.worldScale;
			push.cloudMinSamples = cloudState.minSamples;
			push.cloudMaxSamples = cloudState.maxSamples;
			push.cloudExtinction = cloudState.extinction;
			push.deltaTime = frameParams.time;
			push.cloudAltitude = cloudState.altitude;
			push.cloudThickness = cloudState.thickness;
			push.cloudDensity = cloudState.density;
			push.cloudCoverage = cloudState.coverage;
			push.cloudExtinctionColor = cloudState.extinctionColor;
			push.cloudAlbedo = cloudState.albedo;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudRenderPushConstants), &push);

			uint32_t tileCols = (ctx.width + 7) / 8;
			uint32_t tileRows = (ctx.height + 7) / 8;
			vkCmd.dispatch(tileCols * tileRows, 1, 1);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudTemporalNode : render::NodeRegistrar<CloudTemporalNode> {
		using Resources = graph::Declares<
			graph::Read<CloudPackedColor>,
			graph::Read<CloudPackedDepth>,
			graph::Read<CloudPackedVelocity>,
			graph::Read<CloudBoundingTexture>,
			graph::Create<CloudTemporalColor>,
			graph::Create<CloudTemporalMoments>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_temporal_reprojection.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTemporalColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR16G16B16A16Sfloat),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudTemporalMoments>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudTemporalColor>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 97, // eR16G16B16A16Sfloat
				},
			});
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudTemporalMoments>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 109, // eR32G32B32A32Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			bool hasHistory = false;
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				auto mgr = ServiceLocator::Instance().Get<ICloudManager>();
				cloudState = mgr->GetState();
				hasHistory = mgr->HasHistory();
			}

			auto* registry = static_cast<graph::PhysicalResourceRegistry*>(ctx.resources);

			CloudTemporalPushConstants push{};
			push.packedColorIdx = ctx.Index<CloudPackedColor>();
			push.packedDepthIdx = ctx.Index<CloudPackedDepth>();
			push.packedVelocityIdx = ctx.Index<CloudPackedVelocity>();
			push.boundingMapIdx = ctx.Index<CloudBoundingTexture>();
			push.colorStorageIdx = ctx.StorageIndex<CloudTemporalColor>();
			push.depthStorageIdx = ctx.StorageIndex<CloudPackedDepth>();
			push.momentsStorageIdx = ctx.StorageIndex<CloudTemporalMoments>();
			push.errorMapStorageIdx = ctx.StorageIndex<CloudErrorMap>();
			push.tileQueueBufferAddr = registry ? registry->GetBufferDeviceAddress<CloudTileQueueSSBO>() : 0;
			push.cloudTemporalGamma = cloudState.temporalGamma;
			push.cloudMaxHistoryLength = cloudState.maxHistoryLength;
			push.cloudMaxRayDistance = cloudState.maxRayDistance;
			push.renderScale = cloudState.renderScale;
			push.deltaTime = frameParams.time;
			push.enableTemporal = cloudState.enableTemporal ? 1 : 0;
			push.hasHistory = hasHistory ? 1 : 0;
			push.useTileQueue = cloudState.enableTileScheduler ? 1 : 0;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudTemporalPushConstants), &push);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
#else
			(void)ctx;
#endif
		}
	};

	struct CloudSpatialFilterNode : render::NodeRegistrar<CloudSpatialFilterNode> {
		using Resources = graph::Declares<
			graph::Read<CloudTemporalColor>,
			graph::Read<CloudPackedDepth>,
			graph::Read<CloudTemporalMoments>,
			graph::Read<CloudErrorMap>,
			graph::Create<CloudFilteredColor>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
		render::NodeFrameParams  frameParams{};
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

		void SetFrameParams(const render::NodeFrameParams& params) { frameParams = params; }

		void Init(const render::NodeServices& services) {
#if BRASSICA_HAS_VULKAN
			pipelineLibrary = services.pipelineLibrary;
			compShader.CompileComputeFromFile(services.device, "shaders/effects/cloud_spatial_filter.comp");
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
#else
			(void)services;
#endif
		}

		void RegisterShaders(ShaderWatcher& watcher) {
#if BRASSICA_HAS_VULKAN
			watcher.RegisterShader(&compShader);
#else
			(void)watcher;
#endif
		}

		void Destroy(vk::Device device) {
#if BRASSICA_HAS_VULKAN
			compShader.Destroy(device);
#else
			(void)device;
#endif
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
#if BRASSICA_HAS_VULKAN
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudFilteredColor>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(ctx.width, ctx.height, vk::Format::eR32G32B32A32Sfloat),
				}
			);
#else
			r.realizations.push_back(graph::ResourceRealization{
				.key = graph::IdOf<CloudFilteredColor>(),
				.access = graph::AccessKind::Write,
				.desc = graph::ResourceDesc{
					.kind = graph::ResourceDesc::Kind::Image2D,
					.width = ctx.width,
					.height = ctx.height,
					.formatCode = 109, // eR32G32B32A32Sfloat
				},
			});
#endif
			return r;
		}

		void Execute(graph::NodeContext& ctx) {
#if BRASSICA_HAS_VULKAN
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout),
				static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout)
			};

			render::ComputePipelineRequest request{.shader = &compShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

			ICloudManager::State cloudState{};
			if (ServiceLocator::Instance().Has<ICloudManager>()) {
				cloudState = ServiceLocator::Instance().Get<ICloudManager>()->GetState();
			}

			CloudSpatialFilterPushConstants push{};
			push.cloudColorIdx = ctx.Index<CloudTemporalColor>();
			push.cloudDepthIdx = ctx.Index<CloudPackedDepth>();
			push.cloudMomentsIdx = ctx.Index<CloudTemporalMoments>();
			push.errorMapIdx = ctx.Index<CloudErrorMap>();
			push.filteredColorStorageIdx = ctx.StorageIndex<CloudFilteredColor>();
			push.stepSize = 1;
			push.passIndex = 0;
			push.phiLuma = cloudState.phiLuma;
			push.phiDensity = cloudState.phiDensity;
			push.phiDepth = cloudState.phiDepth;
			push.svgfHistoryBoost = cloudState.svgfHistoryBoost;
			push.svgfHistoryThreshold = cloudState.svgfHistoryThreshold;

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(resolved.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudSpatialFilterPushConstants), &push);

			uint32_t gx = (ctx.width + 7) / 8;
			uint32_t gy = (ctx.height + 7) / 8;
			vkCmd.dispatch(gx, gy, 1);
#else
			(void)ctx;
#endif
		}
	};

	BRASSICA_REGISTER_NODE(CloudBakeNode);
	BRASSICA_REGISTER_NODE(CloudBoundingNode);
	BRASSICA_REGISTER_NODE(CloudShadowBakeNode);
	BRASSICA_REGISTER_NODE(CloudTileSchedulerNode);
	BRASSICA_REGISTER_NODE(CloudRenderNode);
	BRASSICA_REGISTER_NODE(CloudTemporalNode);
	BRASSICA_REGISTER_NODE(CloudSpatialFilterNode);

} // namespace brassica
