#pragma once

#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"
#include <glm/glm.hpp>

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "lighting/ILightManager.hpp"
#include "passes/RenderPhases.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	struct GTAOPushConstants {
		std::uint32_t gPositionIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t outGtaoIndex{0};
		glm::uvec2    screenSize{0, 0};
		float         radiusWorld{1.0f};
		float         intensity{1.0f};
	};

	struct GTAONode: render::NodeRegistrar<GTAONode> {
		render::PipelineLibrary* pipelineLibrary = nullptr;
		ComputeShader            gtaoShader;
		GTAOPushConstants        push{};
		std::uint32_t            width{0};
		std::uint32_t            height{0};
		bool                     enabled{true};

		void SetFrameParams(const render::NodeFrameParams& p) {
			if (ServiceLocator::Instance().Has<ILightManager>()) {
				auto lightManager = ServiceLocator::Instance().Get<ILightManager>();
				enabled = lightManager->IsGTAOEnabled();
				push.radiusWorld = lightManager->GetGTAORadius();
				push.intensity = lightManager->GetGTAOIntensity();
			}
		}

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!gtaoShader.CompileComputeFromFile(services.device, "shaders/gtao.comp")) {
				spdlog::critical("GTAONode shader compilation failed.");
				throw std::runtime_error("GTAONode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&gtaoShader);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<GTAOTexture>();
			}
		}

		void Destroy(vk::Device device) { gtaoShader.Destroy(device); }

		// S4: no kPhase -- Read<GBufferPosition/Normal/Depth>() restored now that those keys are
		// catalog-converted (batch 2), so the lesson #10 hazard that justified omitting them
		// (and the GIComput phase that worked around it) is gone. The real ordering constraint
		// ("after whichever GBuffer writers actually ran this frame") now comes from these reads
		// as ordinary resource edges instead of a phase number.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext& ctx) {
			width = ctx.width;
			height = ctx.height;

			r.Domain(graph::ExecutionDomain::Compute).Active(enabled);
			r.Read<GBufferPosition>();
			r.Read<GBufferNormal>();
			r.Read<GBufferDepth>();
			r.Create<GTAOTexture>();
		}

		void Execute(graph::NodeContext& ctx) {
			if (width == 0 || height == 0) {
				return;
			}

			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gNormalIndex = ctx.Index<GBufferNormal>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.outGtaoIndex = ctx.StorageIndex<GTAOTexture>();
			push.screenSize = glm::uvec2(width, height);

			render::ResolvedPipeline resolved = render::BindCompute(ctx, *pipelineLibrary, gtaoShader, push);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			std::uint32_t     dispatchX = (width + 7) / 8;
			std::uint32_t     dispatchY = (height + 7) / 8;
			vkCmd.dispatch(dispatchX, dispatchY, 1);
		}
	};

	BRASSICA_REGISTER_NODE(GTAONode);

} // namespace brassica
