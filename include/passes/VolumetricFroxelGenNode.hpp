#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct VolumetricNoiseGenPushConstants {
		std::uint32_t noiseStorageIdx{0xFFFFFFFFu};
		float         time{0.0f};
		float         _pad[2]{0.0f, 0.0f};
	};

	struct VolumetricFroxelGenPushConstants {
		alignas(16) glm::vec4 cameraPos{0.0f};                     // xyz = cameraPos, w = time
		alignas(16) glm::vec4 camForward{0.0f, 0.0f, -1.0f, 1.0f}; // xyz = camForward, w = tanHalfFov
		alignas(16) glm::vec4 camUp{0.0f, 1.0f, 0.0f, 1.777f};     // xyz = camUp, w = aspect
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 1.0f};      // xyz = sunDir, w = sunIntensity
		alignas(16) glm::vec4 sunColor{3.0f, 2.94f, 2.76f, 0.6f};  // rgb = sunColor, w = phase anisotropy g
		alignas(
			16
		) glm::vec4 fogParams{0.003f, 0.003f, 0.0f, 1024.0f}; // x = density, y = falloff, z = base, w = gridTextureDim
		alignas(16) glm::uvec4 indices0{0xFFFFFFFFu}; // x = scat0|1, y = scat2|ext0, z = ext1|2, w = noise|skyView
		alignas(16) glm::uvec4 indices1{0xFFFFFFFFu}; // x = clipmap|minMax, y = cloudShadow, z = 0, w = 0
	};

	static_assert(
		sizeof(VolumetricFroxelGenPushConstants) == 128,
		"VolumetricFroxelGenPushConstants must be exactly 128 bytes"
	);

	// Manages 3 froxel grid cascades for volumetric fog, scattering, and god rays.
	// S4: no kPhase -- Read<SkyViewLUT>()/Read<CloudShadowMap>() already order it after each
	// key's sole Create via ordinary resource edges, so the SubPhase::LightPreparation
	// placement was redundant.
	struct VolumetricFroxelGenNode: render::NodeRegistrar<VolumetricFroxelGenNode> {
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
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<VolumetricCascade0Scattering>();
				catalog.Get<VolumetricCascade0Extinction>();
				catalog.Get<VolumetricCascade1Scattering>();
				catalog.Get<VolumetricCascade1Extinction>();
				catalog.Get<VolumetricCascade2Scattering>();
				catalog.Get<VolumetricCascade2Extinction>();
				catalog.Get<VolumetricNoiseTexture>();
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

		// No Read<TerrainClipmapTexture/TerrainMinMaxTexture>() here, matching this node's own
		// pre-conversion Recipe exactly (it never realized either, despite the removed Resources
		// declaring both) -- and deliberately so now too: neither key is catalog-converted yet
		// (a later S2b batch), and TerrainGenNode's Prepare phase schedules before this node's
		// LightPreparation, so an empty-desc realization here would win via lesson #10's last-
		// writer-wins and corrupt the real terrain descs. See GTAONode.hpp/CloudShadowGenNode.hpp
		// for the same reasoning. CloudShadowMap and TerrainWeatherBiomeTexture, by contrast, are
		// both safe to Read<>: both are catalog-converted (CloudShadowGenNode.hpp/
		// ResourceKeys.hpp), and both are real genuine data dependencies this node's shader
		// already reads (cloudShadowIndex, weatherBiomeIndex) -- declaring them as real edges is
		// what makes Provision/barrier synthesis actually order this node after their producers,
		// instead of relying on nothing.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute);
			r.Create<VolumetricCascade0Scattering>();
			r.Create<VolumetricCascade0Extinction>();
			r.Create<VolumetricCascade1Scattering>();
			r.Create<VolumetricCascade1Extinction>();
			r.Create<VolumetricCascade2Scattering>();
			r.Create<VolumetricCascade2Extinction>();
			r.Create<VolumetricNoiseTexture>();
			r.Read<SkyViewLUT>();
			r.Read<CloudShadowMap>();
			r.Read<TerrainWeatherBiomeTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// 1. Generate 2D noise map
			noisePush.noiseStorageIdx = ctx.StorageIndex<VolumetricNoiseTexture>();
			noisePush.time = time;

			render::ResolvedPipeline noiseResolved = render::BindCompute(ctx, *pipelineLibrary, noiseShader, noisePush);
			if (noiseResolved.pipeline) {
				// 512 / 16 = 32
				vkCmd.dispatch(32, 32, 1);
			}

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
			std::uint32_t weatherBiomeIndex = ctx.Index<TerrainWeatherBiomeTexture>();

			genPush.indices0.x = pack16(scat0, scat1);
			genPush.indices0.y = pack16(scat2, ext0);
			genPush.indices0.z = pack16(ext1, ext2);
			genPush.indices0.w = pack16(noiseTextureIdx, skyViewIndex);

			genPush.indices1.x = pack16(clipmapIndex, minMaxIndex);
			genPush.indices1.y = pack16(cloudShadowIndex, weatherBiomeIndex);
			genPush.indices1.z = 0u;
			genPush.indices1.w = 0u;

			render::ResolvedPipeline genResolved = render::BindCompute(ctx, *pipelineLibrary, genShader, genPush);
			if (genResolved.pipeline) {
				// 128 / 8 = 16, 96 / 8 = 12 workgroups (z is integrated in shader column)
				vkCmd.dispatch(16, 12, 1);
			}
		}
	};

	BRASSICA_REGISTER_NODE(VolumetricFroxelGenNode);

} // namespace brassica
