#pragma once

#include <array>
#include <cstdint>

#include "spdlog/spdlog.h"
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

	// kFoliageArchetypeCount etc. live in ResourceKeys.hpp now, alongside the catalog-converted
	// FoliageArchetypeVertexBuffer/IndexBuffer/MetaBuffer Desc()s that need them too.

	// Bindless storage-buffer indices (ctx.BufferIndex<K>(), binding 6 -- see
	// shaders/model_instance.glsl's own comment), not device addresses.
	struct FoliageBakePushConstants {
		std::uint32_t vertexBufIdx{0};
		std::uint32_t indexBufIdx{0};
		std::uint32_t metaBufIdx{0};
	};

	static_assert(sizeof(FoliageBakePushConstants) == 12, "FoliageBakePushConstants size must be 12 bytes");

	// Bakes a small fixed palette of tree/bush archetype shapes into persistent buffers, once,
	// instead of every visible clump regenerating its own unique tree/bush procedurally every
	// frame (FoliageNode's foliage.mesh still does that for grass/fern/flower/rock/seaweed, and
	// for trees/bushes too until the archetype-consuming render pass replaces those calls -- see
	// shaders/foliage_emit.glsl's own comment). This node's only job is populating the buffers;
	// nothing reads them yet.
	struct FoliageBakeNode: render::NodeRegistrar<FoliageBakeNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eNone,
			.depthTest = false,
			.depthWrite = false,
			.enableShadingRate = false,
		};

		render::PipelineLibrary* pipelineLibrary = nullptr;
		MeshShader                              meshShader;
		FragmentShader                          fragShader;
		const brassica::DispatchLoaderDynamic*  dls = nullptr;
		FoliageBakePushConstants                push{};
		bool                                     hasBaked{false};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!meshShader.CompileMeshFromFile(services.device, "shaders/foliage_bake.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/foliage_bake.frag")) {
				spdlog::critical("FoliageBakeNode shader compilation failed.");
				throw std::runtime_error("FoliageBakeNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				RegisterShaders(*services.shaderWatcher);
			}

			// Required for a catalog-converted key's Desc() to ever actually be consulted --
			// ResourceCatalog::DescFor() only answers for a key once some Init() has touched it
			// via Get<K>() (see ResourceCatalog.hpp's own comment). Without this, these three
			// keys' bare Create<>() calls in Setup() below carry an empty desc that Provision()
			// silently ignores (its own "an empty desc is simply ignored for provisioning
			// purposes" rule, lesson #10/#12a) -- the buffer never gets inserted into the
			// registry at all, surfacing downstream as BarrierTranslator's "resolved to neither a
			// texture, a buffer, nor an acceleration structure" the first time anything tries to
			// barrier it. Confirmed as the real root cause, not hypothetical.
			if (services.physicalRegistry) {
				graph::ResourceCatalog& catalog = services.physicalRegistry->Catalog();
				catalog.Get<FoliageArchetypeVertexBuffer>();
				catalog.Get<FoliageArchetypeIndexBuffer>();
				catalog.Get<FoliageArchetypeMetaBuffer>();
			}
		}

		void RegisterShaders(ShaderWatcher& watcher) {
			watcher.RegisterShader(&meshShader);
			watcher.RegisterShader(&fragShader);
		}

		void Destroy(vk::Device /*device*/) {
			// ComputeShader/GraphicsShader-style Destroy is called by the owning ShaderWatcher
			// registration path for these two; nothing additional owned here.
		}

		// FoliageArchetypeVertexBuffer/IndexBuffer/MetaBuffer are catalog-converted (ResourceKeys.hpp)
		// -- bare Create<>(), no .Desc() here, so there's only ever one source of truth for their
		// size/usage instead of every toucher repeating (and risking disagreeing on) its own.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext& fc) {
			r.Domain(graph::ExecutionDomain::Graphics);
			r.Create<FoliageArchetypeVertexBuffer>();
			r.Create<FoliageArchetypeIndexBuffer>();
			r.Create<FoliageArchetypeMetaBuffer>();

			// Sized to the real frame, not a tiny throwaway extent -- BindGraphics sets the
			// viewport/scissor from ctx.width/ctx.height (the frame's own size), not from this
			// attachment's own extent, so a mismatched tiny target here would set a viewport
			// larger than the render area it's actually bound against.
			std::uint32_t w = fc.width > 0 ? fc.width : 1;
			std::uint32_t h = fc.height > 0 ? fc.height : 1;
			r.Create<FoliageBakeDummyTarget>().Desc(graph::ColorAttachmentDesc(w, h, vk::Format::eR8G8B8A8Unorm));
		}

		void Execute(graph::NodeContext& ctx) {
			// The three buffers are re-asserted via Create<>() every frame (same reason
			// TerrainGenNode's clipmap textures are, see its own comment: ProvisionBuffer's
			// desc-match early return leaves the real allocation untouched), but the actual bake
			// -- the expensive part -- only needs to happen once.
			if (hasBaked) {
				return;
			}

			if (const auto* physReg = dynamic_cast<const graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (!physReg->GetBuffer<FoliageArchetypeVertexBuffer>() ||
				    !physReg->GetBuffer<FoliageArchetypeIndexBuffer>() ||
				    !physReg->GetBuffer<FoliageArchetypeMetaBuffer>()) {
					return;
				}
			} else {
				return;
			}

			push.vertexBufIdx = ctx.BufferIndex<FoliageArchetypeVertexBuffer>();
			push.indexBufIdx = ctx.BufferIndex<FoliageArchetypeIndexBuffer>();
			push.metaBufIdx = ctx.BufferIndex<FoliageArchetypeMetaBuffer>();

			std::array<GraphicsShader*, 2> stages{&meshShader, &fragShader};
			std::array<vk::Format, 1>      colorFormats{vk::Format::eR8G8B8A8Unorm};
			render::ResolvedPipeline        resolved = render::BindGraphics(
                ctx,
                *pipelineLibrary,
                stages,
                kPipelineState,
                colorFormats,
                push,
                vk::ShaderStageFlagBits::eMeshEXT,
                vk::Format::eUndefined
            );
			if (!resolved.pipeline) {
				return;
			}

			vk::CommandBuffer vkCmd(static_cast<VkCommandBuffer>(ctx.cmd.vkCmd));
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(kFoliageArchetypeCount, 1, 1, *dls);
			}

			hasBaked = true;
		}
	};

	BRASSICA_REGISTER_NODE(FoliageBakeNode);

} // namespace brassica
