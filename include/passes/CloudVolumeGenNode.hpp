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
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct CloudVolumeGenPushConstants {
		alignas(16) glm::vec4 cameraPos{0.0f};
		alignas(16) glm::uvec4 cascadeStorageIdx{0xFFFFFFFFu};
		alignas(16) glm::vec4 cascadeExtents{20000.0f, 80000.0f, 320000.0f, 15000.0f};
		alignas(16) glm::vec4 cascadeCenterPos[3]{};
		alignas(16) glm::ivec4 cascadeGridOffset[3]{};
		std::uint32_t forceRegeneration{1};
		float         cadencePhase{0.0f};
	};

	struct CloudVolumeCascadeState {
		glm::vec2  centerPos{0.0f};
		glm::ivec3 gridOffset{0};

		void Update(const glm::vec3& cameraPos, float hExtent, bool forceRegen) {
			float     voxelSize = hExtent / 256.0f;
			glm::vec2 newCenter = glm::floor(glm::vec2(cameraPos.x, cameraPos.z) / voxelSize) * voxelSize;
			glm::vec2 diff = newCenter - centerPos;

			int deltaX = static_cast<int>(std::round(diff.x / voxelSize));
			int deltaZ = static_cast<int>(std::round(diff.y / voxelSize));

			if (forceRegen || deltaX != 0 || deltaZ != 0) {
				centerPos = newCenter;
				gridOffset.x = (gridOffset.x + deltaX) % 256;
				if (gridOffset.x < 0)
					gridOffset.x += 256;
				gridOffset.z = (gridOffset.z + deltaZ) % 256;
				if (gridOffset.z < 0)
					gridOffset.z += 256;
			}
		}
	};

	// Manages 3 toroidal volume texture cascades (256^3 each, R16 sfloat)
	// covering roughly 20km, 80km, and 320km of range respectively with a 15km vertical span.
	// Triggers updates as the camera moves similarly to the terrain clipmap generator.
	struct CloudVolumeGenNode: render::NodeRegistrar<CloudVolumeGenNode> {
		render::PipelineLibrary*    pipelineLibrary = nullptr;
		ComputeShader               genShader;
		CloudVolumeGenPushConstants push{};
		glm::vec3                   cameraPos{0.0f};
		glm::vec3                   previousCameraPos{0.0f};
		bool                        hasUpdate{true};
		bool                        forceRegeneration{true};
		uint32_t                    frameCount{0};

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
			if (services.physicalRegistry) {
				auto& catalog = services.physicalRegistry->Catalog();
				catalog.Get<CloudVolumeCascade0>();
				catalog.Get<CloudVolumeCascade1>();
				catalog.Get<CloudVolumeCascade2>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) { watcher.RegisterShader(&genShader); }

		void Destroy(vk::Device device) { genShader.Destroy(device); }

		void SetFrameParams(const render::NodeFrameParams& p) {
			previousCameraPos = cameraPos;
			cameraPos = p.cameraPosition;
			hasUpdate = (cameraPos.x != previousCameraPos.x || cameraPos.z != previousCameraPos.z);
			forceRegeneration = p.forceRegeneration;
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Compute).Active(true);
			r.Create<CloudVolumeCascade0>();
			r.Create<CloudVolumeCascade1>();
			r.Create<CloudVolumeCascade2>();
		}

		void Execute(graph::NodeContext& ctx) {
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
			push.cadencePhase = static_cast<float>(frameCount / 30) * 0.1f;

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, genShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			// Dispatch 256 / 8 = 32 workgroups per dimension
			vkCmd.dispatch(32, 32, 32);
		}
	};

	BRASSICA_REGISTER_NODE(CloudVolumeGenNode);

} // namespace brassica
