#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
#include <glm/glm.hpp>

#include "foliage/IFoliageManager.hpp"
#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"

namespace brassica {

	// Matches shaders/foliage_instance.glsl's FoliageIndirectArgs layout (and
	// VkDrawMeshTasksIndirectCommandEXT) exactly -- only needed here for its sizeof(), as the
	// indirect draw's stride.
	struct FoliageIndirectArgsLayout {
		std::uint32_t groupCountX;
		std::uint32_t groupCountY;
		std::uint32_t groupCountZ;
	};

	static_assert(sizeof(FoliageIndirectArgsLayout) == 12, "FoliageIndirectArgsLayout size must be 12 bytes");

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6 -- see
	// shaders/model_instance.glsl's own comment), not device addresses.
	struct FoliageArchetypeRenderPushConstants {
		std::uint32_t instanceBufIdx{0};
		std::uint32_t vertexBufIdx{0};
		std::uint32_t indexBufIdx{0};
		std::uint32_t metaBufIdx{0};
		float         lengthMultiplier{1.0f};
		float         widthMultiplier{1.0f};
	};

	static_assert(
		sizeof(FoliageArchetypeRenderPushConstants) == 24,
		"FoliageArchetypeRenderPushConstants size must be 24 bytes"
	);

	// Draws this frame's collected tree/bush instances (FoliageNode's foliage.mesh collects them,
	// see its own comment at the append site) from the baked archetype shapes (FoliageBakeNode)
	// instead of regenerating them. foliage.mesh no longer emits live tree/bush geometry at all
	// (the cutover that removed emitBushItem/emitTreeItem's call sites there) -- this node is now
	// the only thing that draws them.
	struct FoliageArchetypeRenderNode: render::NodeRegistrar<FoliageArchetypeRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess, // matches FoliageNode's own fix -- see its comment
			.enableShadingRate = false,
		};

		render::PipelineLibrary*               pipelineLibrary = nullptr;
		MeshShader                              meshShader;
		FragmentShader                          fragShader;
		const brassica::DispatchLoaderDynamic*  dls = nullptr;
		FoliageArchetypeRenderPushConstants     push{};
		bool                                    enabled{true};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			// foliage.frag is FoliageNode's own fragment shader, reused unchanged -- see
			// foliage_archetype_render.mesh's comment on why its varyings already match.
			if (!meshShader.CompileMeshFromFile(services.device, "shaders/foliage_archetype_render.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/foliage.frag")) {
				spdlog::critical("FoliageArchetypeRenderNode shader compilation failed.");
				throw std::runtime_error("FoliageArchetypeRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device device) {
			meshShader.Destroy(device);
			fragShader.Destroy(device);
		}

		void SetFrameParams(const render::NodeFrameParams& /*p*/) {
			if (ServiceLocator::Instance().Has<IFoliageManager>()) {
				auto mgr = ServiceLocator::Instance().Get<IFoliageManager>();
				enabled = mgr->IsEnabled();
				auto props = mgr->GetGlobalProperties();
				push.lengthMultiplier = props.lengthMultiplier;
				push.widthMultiplier = props.widthMultiplier;
			}
		}

		// Real Read<>() on all five buffers this node consumes via bindless storage-buffer index
		// (ctx.BufferIndex<K>(), binding 6 in a push constant) -- each is catalog-converted
		// (ResourceKeys.hpp), so an un-Desc()'d Read<>() here can never win against
		// FoliageBakeNode's/FoliageNode's own
		// Create<>() regardless of schedule order (the lesson-10 hazard EntityNode.hpp's identical-
		// looking EntityInstanceBuffer/EntityIndirectBuffer still have to dodge via After<>() only
		// applies to non-catalog-converted keys). That makes a real resource edge safe to declare,
		// which is what actually buys the automatic barrier -- the graph-level dependency
		// synthesizes it from these Read<>()/Create<>() declarations alone, independent of how the
		// shader later names the bytes (buffer_reference here, same as a bindless index or a fixed
		// descriptor binding would be). AnyOrder() on the GBuffer keys matches FoliageNode/
		// EntityNode's own identical reasoning (both already Modify these same four keys with
		// nothing else ordering them).
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics).Active(enabled);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.Read<FoliageArchetypeVertexBuffer>();
			r.Read<FoliageArchetypeIndexBuffer>();
			r.Read<FoliageArchetypeMetaBuffer>();
			r.Read<FoliageInstanceBuffer>();
			r.Read<FoliageIndirectArgsBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			vk::Buffer     indirectBuf{};
			vk::DeviceSize indirectOffset = 0;

			auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources);
			if (!physReg || !physReg->GetBuffer<FoliageInstanceBuffer>() ||
			    !physReg->GetBuffer<FoliageArchetypeVertexBuffer>()) {
				return;
			}
			if (auto buf = physReg->GetBuffer<FoliageIndirectArgsBuffer>()) {
				indirectBuf = buf->GetBuffer();
			}
			if (!indirectBuf) {
				return;
			}

			push.instanceBufIdx = ctx.BufferIndex<FoliageInstanceBuffer>();
			push.vertexBufIdx = ctx.BufferIndex<FoliageArchetypeVertexBuffer>();
			push.indexBufIdx = ctx.BufferIndex<FoliageArchetypeIndexBuffer>();
			push.metaBufIdx = ctx.BufferIndex<FoliageArchetypeMetaBuffer>();

			std::array<GraphicsShader*, 2> stages{&meshShader, &fragShader};
			std::array<vk::Format, 3>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb
            };
			render::ResolvedPipeline resolved = render::BindGraphics(
				ctx,
				*pipelineLibrary,
				stages,
				kPipelineState,
				colorFormats,
				push,
				vk::ShaderStageFlagBits::eMeshEXT,
				vk::Format::eD32Sfloat
			);
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));

			// No manual barrier needed here anymore -- the five Read<>() declarations in Setup()
			// above give the graph everything it needs to synthesize the correct Acquire barrier
			// against whatever FoliageBakeNode/FoliageNode last wrote, the same as any other
			// Read<>()'d resource in this codebase.
			if (dls && dls->vkCmdDrawMeshTasksIndirectEXT) {
				vkCmd.drawMeshTasksIndirectEXT(indirectBuf, indirectOffset, 1, sizeof(FoliageIndirectArgsLayout), *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(FoliageArchetypeRenderNode);

} // namespace brassica
