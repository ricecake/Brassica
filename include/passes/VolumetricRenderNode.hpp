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

	// Forward-declared only: After<CloudRenderNode>() below needs just the type's name.
	struct CloudRenderNode;

	struct alignas(16) VolumetricRenderPushConstants {
		alignas(16) glm::vec4 camForward{0.0f, 0.0f, -1.0f, 0.0f};
		alignas(16) glm::vec4 camUp{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 camRight{1.0f, 0.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 fovAspect{1.0f, 1.777f, 0.0f, 0.0f}; // tanHalfFov, aspect
		alignas(16) glm::uvec4 cascadeScatteringIdx{0xFFFFFFFFu};
		alignas(16) glm::uvec4 cascadeExtinctionIdx{0xFFFFFFFFu};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t hdrColorIndex{0};
		std::uint32_t skyViewIndex{0};
		std::uint32_t transmittanceIndex{0};
		float         maxDistance{100000.0f};
		float         _pad[2]{0.0f, 0.0f};
	};

	// Composites froxel volumetric scattering and extinction onto HdrColor.
	// S4: no kPhase -- After<AtmosphereCompositeNode>()/Before<WaterNode>() (below) place it in
	// the HdrColor writer chain as real edges instead.
	struct VolumetricRenderNode: render::NodeRegistrar<VolumetricRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*      pipelineLibrary = nullptr;
		VertexShader                  vertShader;
		FragmentShader                fragShader;
		VolumetricRenderPushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/volumetric_render.frag")) {
				spdlog::critical("VolumetricRenderNode shader compilation failed.");
				throw std::runtime_error("VolumetricRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
			if (services.physicalRegistry) {
				services.physicalRegistry->Catalog().Get<HdrColor>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&vertShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			vertShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& p) {
			push.camForward = glm::vec4(p.cameraForward, 0.0f);
			push.camUp = glm::vec4(p.cameraUp, 0.0f);
			push.camRight = glm::vec4(p.cameraRight, 0.0f);
			push.fovAspect = glm::vec4(std::tan(p.fov * 0.5f), p.aspectRatio, 0.0f, 0.0f);
		}

		// No Read<GBufferNormal/Albedo/Material>() -- Execute below only indexes Position and
		// Depth; the removed Resources claimed the whole GBuffer<Read> group (another lesson #2
		// over-claim, same shape as AtmosphereCompositeNode.hpp's). After<AtmosphereCompositeNode>()
		// is the other half of that node's Before<VolumetricRenderNode>() -- see its comment.
		// Before<WaterNode>() (S4) is the other half of WaterNode's new After<VolumetricRenderNode>()
		// -- these five HdrColor writers (Deferred/Atmosphere/Volumetric/Water/Cirrus/CloudRender)
		// previously relied entirely on phase-int ordering for every pair except
		// Atmosphere<->Volumetric and Cirrus<->CloudRender; this closes the Volumetric<->Water gap
		// so the real order (S0's recorded schedule: Atmosphere, Volumetric, Water, Cirrus,
		// CloudRender) is a real edge, not an accident of phase numbers, ahead of removing kPhase
		// from this whole cluster.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<GBufferPosition>();
			r.Read<GBufferDepth>();
			r.Read<VolumetricCascade0Scattering>();
			r.Read<VolumetricCascade0Extinction>();
			r.Read<VolumetricCascade1Scattering>();
			r.Read<VolumetricCascade1Extinction>();
			r.Read<VolumetricCascade2Scattering>();
			r.Read<VolumetricCascade2Extinction>();
			r.Read<SkyViewLUT>();
			r.Read<TransmittanceLUT>();
			r.Modify<HdrColor>();
			r.After<CloudRenderNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.cascadeScatteringIdx.x = ctx.Index<VolumetricCascade0Scattering>();
			push.cascadeScatteringIdx.y = ctx.Index<VolumetricCascade1Scattering>();
			push.cascadeScatteringIdx.z = ctx.Index<VolumetricCascade2Scattering>();

			push.cascadeExtinctionIdx.x = ctx.Index<VolumetricCascade0Extinction>();
			push.cascadeExtinctionIdx.y = ctx.Index<VolumetricCascade1Extinction>();
			push.cascadeExtinctionIdx.z = ctx.Index<VolumetricCascade2Extinction>();

			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gDepthIndex = ctx.Index<GBufferDepth>();
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.skyViewIndex = ctx.Index<SkyViewLUT>();
			push.transmittanceIndex = ctx.Index<TransmittanceLUT>();

			std::array<GraphicsShader*, 2> stages{&vertShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR16G16B16A16Sfloat};
			render::ResolvedPipeline       resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eFragment
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			vkCmd.draw(3, 1, 0, 0);
		}
	};

	BRASSICA_REGISTER_NODE(VolumetricRenderNode);

} // namespace brassica
