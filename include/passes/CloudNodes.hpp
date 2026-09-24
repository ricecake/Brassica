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
	#include "graph/PhysicalResource.hpp"
	#include "render/PipelineLibrary.hpp"
	#include "Shader.hpp"
	#include "ShaderWatcher.hpp"
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
#if BRASSICA_HAS_VULKAN
		ComputeShader weatherShader;
		ComputeShader volumeShader;
#endif

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
					.desc = graph::ComputeStorageImageDesc(128, 128, vk::Format::eR16G16B16A16Sfloat),
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

			render::ComputePipelineRequest reqW{.shader = &weatherShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resW = pipelineLibrary->ResolveCached(reqW);

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (resW.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resW.pipeline);
			}

			std::array<vk::DescriptorSet, 2> boundSets{
				static_cast<VkDescriptorSet>(ctx.frameSet),
				static_cast<VkDescriptorSet>(ctx.globalSet)
			};
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resW.layout, 0, boundSets, nullptr);
			}

			vkCmd.dispatch(64, 64, 1);

			render::ComputePipelineRequest reqV{.shader = &volumeShader, .setLayouts = setLayouts};
			render::ResolvedPipeline resV = pipelineLibrary->ResolveCached(reqV);
			if (resV.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resV.pipeline);
			}
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
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudShadowMap>(),
					.access = graph::AccessKind::Write,
					.desc = graph::ComputeStorageImageDesc(512, 512, vk::Format::eR16Sfloat, 8),
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
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
					.desc = graph::BufferDesc{
						.sizeBytes = (tileCols * tileRows + 16) * sizeof(uint32_t) * 2,
						.usageMask = static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eStorageBuffer),
					},
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudIndirectDispatchSSBO>(),
					.access = graph::AccessKind::Write,
					.desc = graph::BufferDesc{
						.sizeBytes = 3 * sizeof(uint32_t),
						.usageMask = static_cast<std::uint32_t>(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer),
					},
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
			graph::Read<CloudTileQueueSSBO>,
			graph::Read<CloudIndirectDispatchSSBO>,
			graph::Create<CloudPackedColor>,
			graph::Create<CloudPackedDepth>,
			graph::Create<CloudPackedVelocity>>;

		static constexpr graph::Phase kPhase = SubPhase::Atmosphere;

		render::PipelineLibrary* pipelineLibrary = nullptr;
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
#if BRASSICA_HAS_VULKAN
		ComputeShader compShader;
#endif

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
