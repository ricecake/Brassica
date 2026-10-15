#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "vulkan/vulkan.hpp"
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
#include "spdlog/spdlog.h"

namespace brassica {

	struct VolumetricNoiseGenPushConstants {
		std::uint32_t noiseStorageIdx{0xFFFFFFFFu};
		float         time{0.0f};
		float         _pad[2]{0.0f, 0.0f};
	};

	struct VolumetricFroxelGenPushConstants {
		alignas(16) glm::vec4 cameraPos{0.0f};             // xyz = cameraPos, w = time
		alignas(16) glm::vec4 camForward{0.0f, 0.0f, -1.0f, 1.0f}; // xyz = camForward, w = tanHalfFov
		alignas(16) glm::vec4 camUp{0.0f, 1.0f, 0.0f, 1.777f};      // xyz = camUp, w = aspect
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 1.0f};     // xyz = sunDir, w = sunIntensity
		alignas(16) glm::vec4 sunColor{3.0f, 2.94f, 2.76f, 0.6f}; // rgb = sunColor, w = phase anisotropy g
		alignas(16) glm::vec4 fogParams{0.003f, 0.003f, 0.0f, 1024.0f}; // x = density, y = falloff, z = base, w = gridTextureDim
		alignas(16) glm::uvec4 indices0{0xFFFFFFFFu};      // x = scat0|1, y = scat2|ext0, z = ext1|2, w = noise|skyView
		alignas(16) glm::uvec4 indices1{0xFFFFFFFFu};      // x = clipmap|minMax, y = cloudShadow, z = 0, w = 0
	};
	static_assert(sizeof(VolumetricFroxelGenPushConstants) == 128, "VolumetricFroxelGenPushConstants must be exactly 128 bytes");

	// Manages 3 froxel grid cascades for volumetric fog, scattering, and god rays.
	// Operates in SubPhase::LightPreparation (Phase 500).
	struct VolumetricFroxelGenNode: render::NodeRegistrar<VolumetricFroxelGenNode> {
		using Resources = graph::Declares<
			graph::Create<VolumetricCascade0Scattering>,
			graph::Create<VolumetricCascade0Extinction>,
			graph::Create<VolumetricCascade1Scattering>,
			graph::Create<VolumetricCascade1Extinction>,
			graph::Create<VolumetricCascade2Scattering>,
			graph::Create<VolumetricCascade2Extinction>,
			graph::Create<VolumetricNoiseTexture>,
			graph::Read<SkyViewLUT>,
			graph::Read<TerrainClipmapTexture>,
			graph::Read<TerrainMinMaxTexture>,
			graph::Read<CloudShadowMap>>;

		static constexpr graph::Phase kPhase = SubPhase::LightPreparation;

		render::PipelineLibrary*         pipelineLibrary = nullptr;
		ComputeShader                    noiseShader;
		ComputeShader                    genShader;
		VolumetricNoiseGenPushConstants  noisePush{};
		VolumetricFroxelGenPushConstants genPush{};
		glm::vec3                        cameraPos{0.0f};
		float                            time{0.0f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!noiseShader.CompileComputeFromFile(services.device, "shaders/volumetric_noise_gen.comp") ||
			    !genShader.CompileComputeFromFile(services.device, "shaders/volumetric_froxel_gen.comp")) {
				spdlog::critical("VolumetricFroxelGenNode shader compilation failed.");
				throw std::runtime_error("VolumetricFroxelGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&noiseShader);
			watcher.RegisterShader(&genShader);
		}

		void Destroy(vk::Device device) {
			noiseShader.Destroy(device);
			genShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPos = p.cameraPosition;
			time = p.time;

			genPush.cameraPos = glm::vec4(cameraPos, time);
			genPush.camForward = glm::vec4(p.cameraForward, std::tan(p.fov * 0.5f));
			genPush.camUp = glm::vec4(p.cameraUp, p.aspectRatio);
			genPush.sunDir = glm::vec4(p.sunDir, 1.0f);
			genPush.sunColor = glm::vec4(p.sunRadiance, 0.6f); // w = anisotropy g
			genPush.fogParams = glm::vec4(0.003f, 0.003f, 0.0f, static_cast<float>(p.terrainGridParams.w));
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(7);

			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade0Scattering>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade0Extinction>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade1Scattering>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade1Extinction>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade2Scattering>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricCascade2Extinction>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricFroxelDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<VolumetricNoiseTexture>(),
					.access = graph::AccessKind::Write,
					.desc = VolumetricNoiseDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<SkyViewLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(192, 108, vk::Format::eR32G32B32A32Sfloat),
				}
			);

			return r;
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<vk::DescriptorSetLayout, 2> setLayouts{
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.frameSetLayout)),
				vk::DescriptorSetLayout(static_cast<VkDescriptorSetLayout>(ctx.globalSetLayout))
			};
			std::array<vk::DescriptorSet, 2> boundSets{
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.frameSet)),
				vk::DescriptorSet(static_cast<VkDescriptorSet>(ctx.globalSet))
			};

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// 1. Generate 2D noise map
			noisePush.noiseStorageIdx = ctx.StorageIndex<VolumetricNoiseTexture>();
			noisePush.time = time;

			std::array<vk::PushConstantRange, 1> noisePushRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(VolumetricNoiseGenPushConstants)}
			};

			render::ComputePipelineRequest noiseRequest{
				.shader = &noiseShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = noisePushRanges,
			};
			render::ResolvedPipeline noiseResolved = pipelineLibrary->ResolveCached(noiseRequest);

			if (noiseResolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, noiseResolved.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, noiseResolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(
				noiseResolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(VolumetricNoiseGenPushConstants),
				&noisePush
			);

			// 512 / 16 = 32
			vkCmd.dispatch(32, 32, 1);

			// 2. Generate froxel cascade volumes
			auto pack16 = [](std::uint32_t low, std::uint32_t high) -> std::uint32_t {
				return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16u);
			};

			std::uint32_t scat0 = ctx.StorageIndex<VolumetricCascade0Scattering>();
			std::uint32_t scat1 = ctx.StorageIndex<VolumetricCascade1Scattering>();
			std::uint32_t scat2 = ctx.StorageIndex<VolumetricCascade2Scattering>();

			std::uint32_t ext0 = ctx.StorageIndex<VolumetricCascade0Extinction>();
			std::uint32_t ext1 = ctx.StorageIndex<VolumetricCascade1Extinction>();
			std::uint32_t ext2 = ctx.StorageIndex<VolumetricCascade2Extinction>();

			std::uint32_t noiseTextureIdx = ctx.Index<VolumetricNoiseTexture>();
			std::uint32_t skyViewIndex = ctx.Index<SkyViewLUT>();
			std::uint32_t clipmapIndex = ctx.Index<TerrainClipmapTexture>();
			std::uint32_t minMaxIndex = ctx.Index<TerrainMinMaxTexture>();
			std::uint32_t cloudShadowIndex = ctx.Index<CloudShadowMap>();

			genPush.indices0.x = pack16(scat0, scat1);
			genPush.indices0.y = pack16(scat2, ext0);
			genPush.indices0.z = pack16(ext1, ext2);
			genPush.indices0.w = pack16(noiseTextureIdx, skyViewIndex);

			genPush.indices1.x = pack16(clipmapIndex, minMaxIndex);
			genPush.indices1.y = pack16(cloudShadowIndex, 0u);
			genPush.indices1.z = 0u;
			genPush.indices1.w = 0u;

			std::array<vk::PushConstantRange, 1> genPushRanges{
				vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(VolumetricFroxelGenPushConstants)}
			};

			render::ComputePipelineRequest genRequest{
				.shader = &genShader,
				.setLayouts = setLayouts,
				.pushConstantRanges = genPushRanges,
			};
			render::ResolvedPipeline genResolved = pipelineLibrary->ResolveCached(genRequest);

			if (genResolved.pipeline) {
				vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, genResolved.pipeline);
			}
			if (boundSets[0] && boundSets[1]) {
				vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, genResolved.layout, 0, boundSets, nullptr);
			}

			vkCmd.pushConstants(
				genResolved.layout,
				vk::ShaderStageFlagBits::eCompute,
				0,
				sizeof(VolumetricFroxelGenPushConstants),
				&genPush
			);

			// 64 / 8 = 8 workgroups per dimension x and y (z is integrated in shader column)
			vkCmd.dispatch(8, 8, 1);
		}
	};

	BRASSICA_REGISTER_NODE(VolumetricFroxelGenNode);

} // namespace brassica
