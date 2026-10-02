#pragma once

#include <array>
#include <cmath>
#include <cstdint>

#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Declaration.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalResource.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "spdlog/spdlog.h"

namespace brassica {

	struct CloudVolumeGenPushConstants {
		alignas(16) glm::vec4 cameraPos{0.0f};
		alignas(16) glm::uvec4 cascadeStorageIdx{0xFFFFFFFFu};
		alignas(16) glm::vec4 cascadeExtents{20000.0f, 80000.0f, 320000.0f, 15000.0f};
		alignas(16) glm::vec4 cascadeCenterPos[3]{};
		alignas(16) glm::ivec4 cascadeGridOffset[3]{};
		alignas(4) std::uint32_t forceRegeneration{1};
		alignas(4) std::uint32_t padding[3]{0};
	};
	static_assert(sizeof(CloudVolumeGenPushConstants) == 160, "CloudVolumeGenPushConstants size must be 160 bytes");

	struct CloudVolumeCascadeState {
		glm::vec2 centerPos{0.0f};
		glm::ivec3 gridOffset{0};

		void Update(const glm::vec3& cameraPos, float hExtent, bool forceRegen) {
			float voxelSize = hExtent / 256.0f;
			glm::vec2 newCenter = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / voxelSize) * voxelSize;
			glm::vec2 diff = newCenter - centerPos;

			int deltaX = static_cast<int>(std::round(diff.x / voxelSize));
			int deltaZ = static_cast<int>(std::round(diff.y / voxelSize));

			if (forceRegen || deltaX != 0 || deltaZ != 0) {
				centerPos = newCenter;
				gridOffset.x = (gridOffset.x + deltaX) % 256;
				if (gridOffset.x < 0) gridOffset.x += 256;
				gridOffset.z = (gridOffset.z + deltaZ) % 256;
				if (gridOffset.z < 0) gridOffset.z += 256;
			}
		}
	};

	// Manages 3 toroidal volume texture cascades (256^3 each, R16 sfloat)
	// covering roughly 20km, 80km, and 320km of range respectively with a 15km vertical span.
	// Triggers updates as the camera moves similarly to the terrain clipmap generator.
	struct CloudVolumeGenNode: render::NodeRegistrar<CloudVolumeGenNode> {
		using Resources = graph::Declares<
			graph::Create<CloudVolumeCascade0>,
			graph::Create<CloudVolumeCascade1>,
			graph::Create<CloudVolumeCascade2>>;

		render::PipelineLibrary*    pipelineLibrary = nullptr;
		ComputeShader               genShader;
		CloudVolumeGenPushConstants push{};
		glm::vec3                   cameraPos{0.0f};
		glm::vec3                   previousCameraPos{0.0f};
		bool                        hasUpdate{true};
		bool                        forceRegeneration{true};

		CloudVolumeCascadeState cascades[3]{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!genShader.CompileComputeFromFile(services.device, "shaders/cloud_volume_gen.comp")) {
				spdlog::critical("CloudVolumeGenNode shader compilation failed.");
				throw std::runtime_error("CloudVolumeGenNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&genShader);
		}

		void Destroy(vk::Device device) {
			genShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			previousCameraPos = cameraPos;
			cameraPos = p.cameraPosition;
			hasUpdate = (cameraPos.x != previousCameraPos.x || cameraPos.z != previousCameraPos.z);
			forceRegeneration = p.forceRegeneration;
		}

		graph::Recipe Setup(const graph::FrameContext& ctx) {
			(void)ctx;
			graph::Recipe r{.domain = graph::ExecutionDomain::Compute};
			r.realizations.reserve(3);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade0>(),
					.access = graph::AccessKind::Write,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade1>(),
					.access = graph::AccessKind::Write,
					.desc = CloudVolumeDesc(),
				}
			);
			r.realizations.push_back(
				graph::ResourceRealization{
					.key = graph::IdOf<CloudVolumeCascade2>(),
					.access = graph::AccessKind::Write,
					.desc = CloudVolumeDesc(),
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

			if (hasUpdate || forceRegeneration) {
				// Update cascade states
				float extents[3] = {push.cascadeExtents.x, push.cascadeExtents.y, push.cascadeExtents.z};
				for (int i = 0; i < 3; ++i) {
					cascades[i].Update(cameraPos, extents[i], forceRegeneration);
					push.cascadeCenterPos[i] = glm::vec4(cascades[i].centerPos.x, 0.0f, cascades[i].centerPos.y, 0.0f);
					push.cascadeGridOffset[i] = glm::ivec4(cascades[i].gridOffset, 0);
				}

				push.cascadeStorageIdx.x = ctx.StorageIndex<CloudVolumeCascade0>();
				push.cascadeStorageIdx.y = ctx.StorageIndex<CloudVolumeCascade1>();
				push.cascadeStorageIdx.z = ctx.StorageIndex<CloudVolumeCascade2>();
				push.cameraPos = glm::vec4(cameraPos, 1.0f);
				push.forceRegeneration = forceRegeneration ? 1 : 0;

				std::array<vk::PushConstantRange, 1> pushConstantRanges{
					vk::PushConstantRange{vk::ShaderStageFlagBits::eCompute, 0, sizeof(CloudVolumeGenPushConstants)}
				};

				render::ComputePipelineRequest request{
					.shader = &genShader,
					.setLayouts = setLayouts,
					.pushConstantRanges = pushConstantRanges,
				};
				render::ResolvedPipeline resolved = pipelineLibrary->ResolveCached(request);

				vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
				if (resolved.pipeline) {
					vkCmd.bindPipeline(vk::PipelineBindPoint::eCompute, resolved.pipeline);
				}

				if (boundSets[0] && boundSets[1]) {
					vkCmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, resolved.layout, 0, boundSets, nullptr);
				}

				vkCmd.pushConstants(
					resolved.layout,
					vk::ShaderStageFlagBits::eCompute,
					0,
					sizeof(CloudVolumeGenPushConstants),
					&push
				);

				// Dispatch 256 / 8 = 32 workgroups per dimension
				vkCmd.dispatch(32, 32, 32);
			}
		}
	};

	BRASSICA_REGISTER_NODE(CloudVolumeGenNode);

} // namespace brassica
