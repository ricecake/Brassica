#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include "vulkan/vulkan.hpp"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/PhysicalResource.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/LightningAgeNode.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "types/LightningStrike.hpp"
#include "VulkanCompat.hpp"

namespace brassica {

	// Forward-declared only: Before<CloudRenderNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct CloudRenderNode;

	// No fields needed: camera/time (FrameUBO) and the strike pool (LightningStrikePool) are
	// both already globally bound (set 0), so there's nothing per-dispatch to push -- but
	// BindGraphics' push-constant range can't be zero-sized, hence the one dummy field.
	struct LightningRenderPushConstants {
		std::uint32_t _unused{0};
	};

	// Draws every active LightningStrikePool slot as camera-facing quad geometry (lightning.task
	// plots the polyline and culls dead slots; lightning.mesh builds the quads; lightning.frag
	// is a flat flicker-tinted unlit output). Before<CloudRenderNode>(): strikes are cloud-to-
	// cloud only (v1 scope, see climate_ca.comp's trigger comment), so drawing before clouds
	// lets CloudRenderNode's own transmittance attenuate the bolt by whatever cloud density sits
	// between it and the camera -- the physically-motivated side of "embedded in the cloud
	// layer," rather than always rendering fully bright on top.
	struct LightningRenderNode: render::NodeRegistrar<LightningRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = false,
			.enableBlend = true,
		};

		render::PipelineLibrary*     pipelineLibrary = nullptr;
		const DispatchLoaderDynamic* dls = nullptr;
		TaskShader                   taskShader;
		MeshShader                   meshShader;
		FragmentShader               fragShader;
		LightningRenderPushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!taskShader.CompileTaskFromFile(services.device, "shaders/lightning.task") ||
			    !meshShader.CompileMeshFromFile(services.device, "shaders/lightning.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/lightning.frag")) {
				spdlog::critical("LightningRenderNode shader compilation failed.");
				throw std::runtime_error("LightningRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&taskShader);
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			taskShader.Destroy(device);
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<LightningStrikePool>();
			r.Read<GBufferDepth>();
			r.Modify<HdrColor>();
			r.After<LightningAgeNode>();
			r.Before<CloudRenderNode>();
			r.Before<CirrusNode>();
			r.Before<WaterNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::array<GraphicsShader*, 3> stages{&taskShader, &meshShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			render::ResolvedPipeline       resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT,
                vk::Format::eD32Sfloat
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(kMaxLightningStrikes, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(LightningRenderNode);

} // namespace brassica
