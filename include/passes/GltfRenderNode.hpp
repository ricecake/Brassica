#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>
#include "spdlog/spdlog.h"

#include "graph/Execution.hpp"
#include "graph/PhysicalRegistry.hpp"
#include "graph/SetupBuilder.hpp"
#include "model/GltfModelManager.hpp"
#include "passes/ResourceKeys.hpp"
#include "render/NodeCommands.hpp"
#include "render/NodeLifecycle.hpp"
#include "render/PipelineLibrary.hpp"
#include "ServiceLocator.hpp"
#include "Shader.hpp"
#include "ShaderWatcher.hpp"
#include "terrain/TerrainManager.hpp"

namespace brassica {

	struct GltfRenderPushConstants {
		std::uint64_t vertexBufAddress{0};
		std::uint64_t meshletVertexBufAddress{0};
		std::uint64_t meshletTriangleBufAddress{0};
		std::uint64_t meshletBufAddress{0};
		// Stage 4 (GPU skinning). Grouped with the other buffer addresses, not appended after the
		// smaller fields below -- feedback_push_constant_struct_editing's "append, don't interleave"
		// rule is about not reshuffling *existing* fields, not about where a new field of a
		// different kind has to land; keeping all five addresses contiguous matches every other
		// GLTF push-constant struct's own shape (see GltfModelManager's buffer members).
		std::uint64_t jointMatrixBufAddress{0};
		// GLSL/std430 pads a vec3 to start at a 16-byte-aligned offset -- 5 uint64s land
		// relModelPosition at byte 40 in a naive C++ layout, but GLSL's own push-constant block
		// (verified via spirv-dis, not hand-computed) puts it at byte 48. These two uint32s make
		// the C++ struct match that real layout instead of silently drifting from it.
		std::uint32_t _pad0{0};
		std::uint32_t _pad1{0};
		glm::vec3     relModelPosition{0.0f}; // model position relative to camera, see Execute()
		float         modelScale{1.0f};
		// Appended, not interleaved (see feedback_push_constant_struct_editing): which bindless
		// slot MaterialBuffer landed in this frame -- gltf_render.mesh forwards it to gltf.frag
		// as a flat varying (push constants are mesh-stage-only in this engine's convention).
		std::uint32_t materialBufferIndex{0};
	};

	// 72: 5 uint64s (40) + the 2 explicit padding uint32s (8, bringing relModelPosition to the
	// real GLSL-verified offset 48) + vec3 (12, to 60) + float (4, to 64) + uint32 (4, to 68),
	// rounded up to the struct's own 8-byte alignment. Verified via sizeof(), not hand-computed
	// from the member list alone -- that's exactly the kind of mistake this assert exists to catch.
	static_assert(sizeof(GltfRenderPushConstants) == 72, "GltfRenderPushConstants size must be 72 bytes");

	// Stages 1-4 of the GLTF/GLB pipeline: draws GltfModelManager's single loaded model from its
	// meshlet buffers, one mesh-shader workgroup per meshlet (see gltf_render.mesh's own comment
	// on why there's no task-shader stage yet), with real glTF materials (baseColor/metallic/
	// roughness/occlusion, textured or flat) via the shared MaterialBuffer, and GPU skinning
	// (gltf_render.mesh blends each vertex's JOINTS_0/WEIGHTS_0 against the joint matrix buffer
	// GltfModelManager::UpdatePose recomputes every frame). modelWorldPosition/modelScale are
	// plain member fields with no manager or UI hook -- the goal so far is proving the
	// load -> GPU buffers -> render path works, not a scene-placement system.
	struct GltfRenderNode: render::NodeRegistrar<GltfRenderNode> {
		static constexpr render::GraphicsPipelineState kPipelineState{
			.cullMode = vk::CullModeFlagBits::eBack,
			.depthTest = true,
			.depthWrite = true,
			.depthCompareOp = vk::CompareOp::eLess,
			.enableShadingRate = false,
		};

		render::PipelineLibrary*               pipelineLibrary = nullptr;
		MeshShader                             meshShader;
		FragmentShader                         fragShader;
		const brassica::DispatchLoaderDynamic* dls = nullptr;
		glm::vec3                              cameraPosition{0.0f};
		// X/Z are a placeholder (world origin) -- no scene-placement system yet, see this
		// struct's own comment. Y is NOT a placeholder: it gets overwritten every frame in
		// SetFrameParams() from the live terrain height at the origin. A hardcoded Y=0 here
		// put the model wherever world-Y=0 happened to be relative to the *actual* ground --
		// invisible (buried or stranded far below camera) the moment terrain height at the
		// origin isn't close to 0, confirmed on real hardware with terrain ~4000 units tall at
		// the spawn point. Re-querying every frame (not once at load) avoids the same kind of
		// stale-snapshot trap a one-time query would have, matching how TerrainManager's own
		// readback-backed cache is meant to be read.
		glm::vec3 modelWorldPosition{0.0f, 0.0f, 0.0f};
		float     modelScale{1.0f};

		void Init(const render::NodeServices& services) {
			pipelineLibrary = services.pipelineLibrary;
			dls = services.dispatchLoader;
			if (!meshShader.CompileMeshFromFile(services.device, "shaders/gltf_render.mesh") ||
			    !fragShader.CompileFragmentFromFile(services.device, "shaders/gltf.frag")) {
				spdlog::critical("GltfRenderNode shader compilation failed.");
				throw std::runtime_error("GltfRenderNode shader compilation failed.");
			}
			if (services.shaderWatcher) {
				services.shaderWatcher->RegisterShader(&meshShader);
				services.shaderWatcher->RegisterShader(&fragShader);
			}
		}

		void Destroy(vk::Device /*device*/) {}

		void SetFrameParams(const render::NodeFrameParams& p) {
			cameraPosition = p.cameraPosition;
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<TerrainManager>()) {
				auto terrain = ServiceLocator::Instance().Get<TerrainManager>();
				// +1.0: an arbitrary small lift, not a real "feet on the ground" placement --
				// cow.glb's own local origin/bounding box isn't accounted for (no scene-placement
				// system yet). Good enough to make the model visible and roughly ground-level;
				// not good enough to claim it never clips into terrain.
				modelWorldPosition.y = terrain->GetCachedGroundHeight(0.0f) + 1.0f;
			}
		}

		[[nodiscard]] static std::uint32_t QueryMeshletCount() {
			if (ServiceLocator::HasInstance() && ServiceLocator::Instance().Has<GltfModelManager>()) {
				return ServiceLocator::Instance().Get<GltfModelManager>()->GetMeshletCount();
			}
			return 0;
		}

		// Bare Read<>() on all four buffers: each has exactly one real producer
		// (GltfModelManager's PredefinedBufferNode members), so no .Desc() belongs here -- see
		// ResourceKeys.hpp's comment on GltfVertexBuffer et al.
		void Setup(graph::SetupBuilder& r, const graph::FrameContext&) {
			r.Domain(graph::ExecutionDomain::Graphics).Active(QueryMeshletCount() > 0);
			r.Modify<GBufferPosition>().AnyOrder();
			r.Modify<GBufferNormal>().AnyOrder();
			r.Modify<GBufferAlbedo>().AnyOrder();
			// gltf.frag writes real metallic/roughness via evaluateGltfMetallicRoughness -- a real
			// GBuffer output, not an unused one (confirmed missing on real hardware: GPU-AV flagged
			// "writes to output Location 3 but there is no pColorAttachments[3]" -- the shader
			// always had this output, the pipeline/realization just never declared a slot for it).
			r.Modify<GBufferMaterial>().AnyOrder();
			r.Modify<GBufferDepth>().AnyOrder();
			r.Read<GltfVertexBuffer>();
			r.Read<GltfMeshletVertexBuffer>();
			r.Read<GltfMeshletTriangleBuffer>();
			r.Read<GltfMeshletBuffer>();
			r.Read<GltfJointMatrixBuffer>();
			// Same bare-Read reasoning: MaterialManager's own PredefinedBufferNode is MaterialBuffer's
			// sole producer.
			r.Read<MaterialBuffer>();
		}

		void Execute(graph::NodeContext& ctx) {
			std::uint32_t meshletCount = QueryMeshletCount();
			if (meshletCount == 0) {
				return;
			}

			GltfRenderPushConstants push{};
			push.relModelPosition = modelWorldPosition - cameraPosition;
			push.modelScale = modelScale;
			push.materialBufferIndex = ctx.BufferIndex<MaterialBuffer>();

			if (auto* physReg = dynamic_cast<graph::PhysicalResourceRegistry*>(ctx.resources)) {
				if (auto buf = physReg->GetBuffer<GltfVertexBuffer>()) {
					push.vertexBufAddress =
						physReg->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo(buf->GetBuffer()));
				}
				if (auto buf = physReg->GetBuffer<GltfMeshletVertexBuffer>()) {
					push.meshletVertexBufAddress =
						physReg->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo(buf->GetBuffer()));
				}
				if (auto buf = physReg->GetBuffer<GltfMeshletTriangleBuffer>()) {
					push.meshletTriangleBufAddress =
						physReg->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo(buf->GetBuffer()));
				}
				if (auto buf = physReg->GetBuffer<GltfMeshletBuffer>()) {
					push.meshletBufAddress =
						physReg->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo(buf->GetBuffer()));
				}
				if (auto buf = physReg->GetBuffer<GltfJointMatrixBuffer>()) {
					push.jointMatrixBufAddress =
						physReg->GetDevice().getBufferAddress(vk::BufferDeviceAddressInfo(buf->GetBuffer()));
				}
			}

			if (push.vertexBufAddress == 0 || push.meshletBufAddress == 0 || push.jointMatrixBufAddress == 0) {
				return;
			}

			std::array<GraphicsShader*, 2> stages{&meshShader, &fragShader};
			std::array<vk::Format, 4>      colorFormats{
                vk::Format::eR32G32B32A32Sfloat,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Format::eR8G8B8A8Srgb,
                vk::Format::eR8G8B8A8Unorm
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
			if (dls && dls->vkCmdDrawMeshTasksEXT) {
				vkCmd.drawMeshTasksEXT(meshletCount, 1, 1, *dls);
			}
		}
	};

	BRASSICA_REGISTER_NODE(GltfRenderNode);

} // namespace brassica
