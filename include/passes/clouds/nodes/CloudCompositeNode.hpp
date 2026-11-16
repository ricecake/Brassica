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
#include "passes/volumetrics/resources.hpp"
#include "resources.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Forward-declared only: After<CirrusNode>() below needs just the type's name -- see
	// ClusterLightAssignmentNode.hpp's identical reasoning.
	struct CirrusNode;

	// Merged cloud + volumetric-fog composite (was CloudCompositeNode + VolumetricRenderNode) --
	// see shaders/cloud_composite.frag's own header comment for why merging them is what makes
	// a camera-altitude-aware composite order possible at all. Field order/types here must match
	// cloud_composite.frag's push_constant block exactly (std430-like packing: every vec4/uvec4
	// below needs 16-byte alignment, which is why the scalars before camForward don't need
	// explicit padding -- GLSL and C++ (via alignas(16)) both insert the same gap on their own).
	struct alignas(16) CloudCompositePushConstants {
		std::uint32_t hdrColorIndex{0};
		std::uint32_t cloudColorIndex{0};
		std::uint32_t cloudTransmittanceIndex{0};
		std::uint32_t gPositionIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t cloudMomentsIndex{0};
		alignas(16) glm::vec4 camForward{0.0f, 0.0f, -1.0f, 0.0f};
		alignas(16) glm::vec4 camUp{0.0f, 1.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 camRight{1.0f, 0.0f, 0.0f, 0.0f};
		alignas(16) glm::vec4 fovAspect{1.0f, 1.777f, 0.0f, 0.0f}; // tanHalfFov, aspect
		alignas(16) glm::uvec4 cascadeScatteringIdx{0xFFFFFFFFu};
		alignas(16) glm::uvec4 cascadeExtinctionIdx{0xFFFFFFFFu};
		std::uint32_t skyViewIndex{0};
		float         maxDistance{100000.0f};
	};

	static_assert(
		sizeof(CloudCompositePushConstants) == 144,
		"CloudCompositePushConstants size must be 144 bytes -- must match cloud_composite.frag's push_constant "
		"block exactly"
	);

	// Blends the cloud layer (CloudTemporalNode's accumulated raymarch) and the volumetric fog
	// layer (froxel cascades) into HdrColor in one pass, in whichever order the camera's altitude
	// relative to the cloud band calls for -- see cloud_composite.frag for the actual blend
	// math. Takes over both CloudCompositeNode's and VolumetricRenderNode's old positions in the
	// HdrColor writer chain; every Before<>/After<> edge elsewhere that targeted either of those
	// two names now only needs CloudCompositeNode (VolumetricRenderNode no longer exists -- it
	// had zero incoming edges from any other file, only this node's own now-removed self-
	// reference, so nothing else needed updating).
	//
	// No more stencil write: CloudCompositeNode used to stamp GBufferDepth's stencil aspect
	// (classification=2) for real cloud pixels via a discard-gated StencilMode::Write, consumed
	// only by tonemap.frag's TEMPORARY debugVisualizeStencil toggle (never load-bearing for the
	// real exposure blend, which already uses CloudAccumulatedTransmittance directly). That
	// relied on discarding non-cloud fragments to skip the stencil write -- structurally
	// incompatible with this merge, since the fog layer still needs to render for every pixel,
	// cloud or not, so nothing can discard anymore. Removed along with its tonemap.frag/
	// LightingWidget debug-only consumers rather than left half-broken (see those files' own
	// comments).
	struct CloudCompositeNode: render::NodeRegistrar<CloudCompositeNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*    pipelineLibrary = nullptr;
		VertexShader                vertShader; // shaders/atmosphere/sky.vert
		FragmentShader              fragShader; // shaders/cloud_composite.frag
		CloudCompositePushConstants push{};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			if (!vertShader.CompileVertexFromFile(services.device, "shaders/atmosphere/sky.vert") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/cloud_composite.frag")) {
				spdlog::critical("CloudCompositeNode shader compilation failed.");
				throw std::runtime_error("CloudCompositeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
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

		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Read<CloudAccumulatedColor>();
			r.Read<CloudAccumulatedTransmittance>();
			r.Read<CloudMoments>();
			r.Read<GBufferPosition>();
			r.Read<GBufferAlbedo>();
			r.Read<VolumetricCascade0Scattering>();
			r.Read<VolumetricCascade0Extinction>();
			r.Read<VolumetricCascade1Scattering>();
			r.Read<VolumetricCascade1Extinction>();
			r.Read<VolumetricCascade2Scattering>();
			r.Read<VolumetricCascade2Extinction>();
			r.Read<SkyViewLUT>();
			r.Modify<HdrColor>();
			r.After<CirrusNode>();
		}

		void Execute(graph::NodeContext& ctx) {
			push.hdrColorIndex = ctx.Index<HdrColor>();
			push.cloudColorIndex = ctx.Index<CloudAccumulatedColor>();
			push.cloudTransmittanceIndex = ctx.Index<CloudAccumulatedTransmittance>();
			push.gPositionIndex = ctx.Index<GBufferPosition>();
			push.gAlbedoIndex = ctx.Index<GBufferAlbedo>();
			push.cloudMomentsIndex = ctx.Index<CloudMoments>();

			push.cascadeScatteringIdx.x = ctx.Index<VolumetricCascade0Scattering>();
			push.cascadeScatteringIdx.y = ctx.Index<VolumetricCascade1Scattering>();
			push.cascadeScatteringIdx.z = ctx.Index<VolumetricCascade2Scattering>();

			push.cascadeExtinctionIdx.x = ctx.Index<VolumetricCascade0Extinction>();
			push.cascadeExtinctionIdx.y = ctx.Index<VolumetricCascade1Extinction>();
			push.cascadeExtinctionIdx.z = ctx.Index<VolumetricCascade2Extinction>();

			push.skyViewIndex = ctx.Index<SkyViewLUT>();

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

	BRASSICA_REGISTER_NODE(CloudCompositeNode);

} // namespace brassica
