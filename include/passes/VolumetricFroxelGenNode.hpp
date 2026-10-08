#pragma once

#include <array>
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
		alignas(16) glm::vec4 cameraPos{0.0f};
		alignas(16) glm::vec4 sunDir{0.0f, 1.0f, 0.0f, 1.0f};
		alignas(16) glm::vec4 sunColor{3.0f, 2.94f, 2.76f, 1.0f};
		alignas(16) glm::vec4 fogParams{0.05f, 0.01f, 0.0f, 0.6f}; // density, height falloff, base height, anisotropy g
		alignas(16) glm::uvec4 cascadeScatteringIdx{0xFFFFFFFFu};
		alignas(16) glm::uvec4 cascadeExtinctionIdx{0xFFFFFFFFu};
		std::uint32_t noiseTextureIdx{0xFFFFFFFFu};
		std::uint32_t transmittanceLUTIdx{0xFFFFFFFFu};
		std::uint32_t skyViewLUTIdx{0xFFFFFFFFu};
		std::uint32_t _padding{0};
	};

	// Manages 3 froxel grid cascades for volumetric fog and scattering.
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
			graph::Read<TransmittanceLUT>,
			graph::Read<SkyViewLUT>>;

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
			genPush.sunDir = glm::vec4(p.sunDir, 1.0f);
			genPush.sunColor = glm::vec4(p.sunRadiance, 1.0f);
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(9);

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
					.key = graph::IdOf<TransmittanceLUT>(),
					.access = graph::AccessKind::Read,
					.desc = graph::ComputeStorageImageDesc(256, 64, vk::Format::eR32G32B32A32Sfloat),
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
			genPush.cascadeScatteringIdx.x = ctx.StorageIndex<VolumetricCascade0Scattering>();
			genPush.cascadeScatteringIdx.y = ctx.StorageIndex<VolumetricCascade1Scattering>();
			genPush.cascadeScatteringIdx.z = ctx.StorageIndex<VolumetricCascade2Scattering>();

			genPush.cascadeExtinctionIdx.x = ctx.StorageIndex<VolumetricCascade0Extinction>();
			genPush.cascadeExtinctionIdx.y = ctx.StorageIndex<VolumetricCascade1Extinction>();
			genPush.cascadeExtinctionIdx.z = ctx.StorageIndex<VolumetricCascade2Extinction>();

			genPush.noiseTextureIdx = ctx.Index<VolumetricNoiseTexture>();
			genPush.transmittanceLUTIdx = ctx.Index<TransmittanceLUT>();
			genPush.skyViewLUTIdx = ctx.Index<SkyViewLUT>();

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

			// 64 / 8 = 8
			vkCmd.dispatch(8, 8, 8);
		}
	};

	BRASSICA_REGISTER_NODE(VolumetricFroxelGenNode);

} // namespace brassica
