#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	// GPU-driven model instance: a slot in ModelInstanceBuffer, written only on spawn/clip-change/
	// despawn (see model/ModelInstanceManager.hpp) and otherwise read every frame with no CPU
	// involvement -- the clip + startTimeMs pair is compared against FrameUBO's uTimeMs entirely
	// on the GPU. Mirrors shaders/model_instance.glsl's ModelInstance exactly.
	struct ModelInstance {
		glm::vec3     position{0.0f};
		float         scale{1.0f};
		glm::vec4     rotation{0.0f, 0.0f, 0.0f, 1.0f}; // quat xyzw, identity
		std::uint32_t modelIndex{0};                    // index into ModelDescBuffer (model/ModelRegistry.hpp)
		std::uint32_t clipIndex{0xFFFFFFFFu};            // ~0u = rest pose
		std::uint32_t startTimeMs{0};
		std::uint32_t flags{0}; // bit 0: alive, bit 1: loop (else hold last frame), bit 2: live-capable
		// Stage 4: per-instance phase diversity for ModelLiveEvalNode's seed-bend/wind-gust
		// evaluation (model_live_eval.comp) -- assigned once at spawn (ModelInstanceManager), not
		// re-randomized per frame, so a given instance's live sway stays temporally coherent.
		// Appended, not interleaved -- see feedback_push_constant_struct_editing. _pad0..2 bring
		// the real 52-byte size up to std430's own array-stride rounding (vec4 rotation forces
		// 16-byte alignment, so 52 rounds to 64) -- verify via spirv-dis, not just this comment.
		std::uint32_t seed{0};
		std::uint32_t _pad0{0};
		std::uint32_t _pad1{0};
		std::uint32_t _pad2{0};
	};

	static_assert(sizeof(ModelInstance) == 64, "ModelInstance must match model_instance.glsl's 64-byte stride");

	inline constexpr std::uint32_t kModelInstanceAliveBit = 1u << 0;
	inline constexpr std::uint32_t kModelInstanceLoopBit = 1u << 1;
	// Stage 4: a persistent, spawner-set property ("this instance is eligible for live wind
	// eval"), distinct from which mode ModelCullNode actually picks for it this frame (distance/
	// pool-budget dependent, never stored back to the record) -- see the plan's "the record
	// stores capability, not the mode used this frame."
	inline constexpr std::uint32_t kModelInstanceLiveCapableBit = 1u << 2;

	// Stage 5.2: ModelInstanceBuffer is split into two non-overlapping ranges within one shared
	// buffer (the plan's own "no new buffer" requirement) -- [0, kManualInstanceCapacity) for
	// ModelInstanceManager's CPU-driven sparse spawn/despawn (unchanged from Stages 1-4, same
	// 1024 slots as before), and [kDecorSlotBase, kDecorSlotBase + kDecorCapacity) reserved for
	// the GPU-direct decor population (DecorChunkAllocator/DecorChunkGenNode) -- a fixed pool of
	// kDecorMaxActiveChunks chunk-blocks, each kDecorInstancesPerChunk slots, matching the
	// per-tile clump count foliage.mesh's own cellCoord/cellSeed grid already uses (16 clumps/
	// tile), not an arbitrary number. kModelInstanceCapacity is the *total* buffer size (what
	// ModelInstanceBuffer's own Desc(), ModelCullNode's full-table walk, and the GLSL mirror in
	// model_instance.glsl all need) -- ModelInstanceManager's own capacity is a separate, smaller
	// constant specifically so its free-list can never wander into the decor range.
	inline constexpr std::uint32_t kManualInstanceCapacity = 1024u;
	inline constexpr std::uint32_t kDecorInstancesPerChunk = 16u;
	// A 9x9 grid of FoliageNode's own LOD-0 tiles (baseTileSize=16 -- include/passes/
	// FoliageNode.hpp's FoliagePushConstants) centered on the camera's current tile, +margin for
	// boundary churn -- roughly a 70-unit radius, in the same ballpark as the existing foliage
	// LOD falloff range (lodBaseRange=20). Not an arbitrary number: DecorChunkAllocator's own
	// candidate generation targets this same 9x9 coverage.
	inline constexpr std::uint32_t kDecorMaxActiveChunks = 96u;
	inline constexpr std::uint32_t kDecorCapacity = kDecorInstancesPerChunk * kDecorMaxActiveChunks;
	inline constexpr std::uint32_t kDecorSlotBase = kManualInstanceCapacity;
	inline constexpr std::uint32_t kModelInstanceCapacity = kManualInstanceCapacity + kDecorCapacity;
	inline constexpr std::uint32_t kModelClipRestPose = 0xFFFFFFFFu;

	// One slot in ModelInstanceWriteBuffer -- a pending sparse write the scatter shader applies
	// to ModelInstanceBuffer[slot] (model_instance_scatter.comp). The CPU never writes
	// ModelInstanceBuffer directly; this is the only path in or out.
	struct ModelInstanceWrite {
		std::uint32_t  slot{0};
		std::uint32_t  _pad0{0};
		std::uint32_t  _pad1{0};
		std::uint32_t  _pad2{0}; // pads to 16 for record's own alignment, matching GLSL std430
		ModelInstance  record{};
	};

	static_assert(sizeof(ModelInstanceWrite) == 16 + sizeof(ModelInstance), "ModelInstanceWrite layout drifted");

	// One baked animation clip's metadata -- mirrors shaders/model_instance.glsl's
	// ModelClipEntry. paletteBase indexes GltfModel::BakeClips's flat joint-matrix palette in
	// *joint* units (not bytes): frame f's joint j lives at palette[paletteBase + f*jointCount +
	// j]. Bounds are model-local (pre instance-transform), the union over every baked frame's
	// skinned vertex positions -- conservative enough to frustum-cull the whole clip's motion
	// range from one sphere.
	struct ModelClipEntry {
		std::uint32_t paletteBase{0};
		std::uint32_t frameCount{0};
		std::uint32_t durationMs{0};
		// The real bake-time sampling interval (1000/sampleRateHz) -- NOT re-derivable from
		// durationMs/(frameCount-1): frameCount's "+2" padding and the integer truncation in its
		// own computation (GltfModel::BakeClips) make that quotient drift from the true interval,
		// and that drift compounds linearly with frame index. A real, measured bug: re-deriving
		// it that way picked an off-by-several-frames pose by the end of a clip, producing a 20x
        // larger skinned-vertex error than storing this directly. Also pads to 16 for
        // boundsCenter's vec3 alignment, same slot the old unused _pad0 occupied.
		float     frameDtMs{0.0f};
		glm::vec3 boundsCenter{0.0f};
		float     boundsRadius{0.0f};
	};

	static_assert(sizeof(ModelClipEntry) == 32, "ModelClipEntry must match model_instance.glsl's 32-byte stride");

	// Stage 2: one entry per registered model (model/ModelRegistry.hpp), indexed by
	// ModelInstance::modelIndex. The engine's resource-key system is one-physical-buffer-per-type
	// (no per-instance-indexed keys), so multiple models can't each own their own vertex/meshlet/
	// palette/clip-table buffer -- instead ModelRegistry concatenates every model's data into one
	// shared buffer per kind, and this entry's *Base fields are where this model's own chunk
	// starts within each shared buffer. Deliberately NOT rebased into the data itself (no
	// per-element CPU-side rewriting): the shader adds the relevant base once at the point of
	// lookup (model_render.task/mesh), so a model's own vertexOffset/triangleOffset/paletteBase
	// fields stay exactly as its producer authored them (0-based, model-local).
	struct ModelDescEntry {
		glm::vec3     restBoundsCenter{0.0f}; // model-local, union of every rest-pose skinned vertex
		float         restBoundsRadius{0.0f};
		std::uint32_t vertexBase{0};             // into the shared GltfVertexBuffer, in GltfVertex units
		std::uint32_t meshletVertexBase{0};       // into the shared GltfMeshletVertexBuffer, in uint32 units
		std::uint32_t meshletTriangleWordBase{0}; // into the shared GltfMeshletTriangleBuffer, in uint32 (word) units
		std::uint32_t meshletBase{0};              // into the shared GltfMeshletBuffer, in GltfMeshlet units
		std::uint32_t meshletCount{0};
		std::uint32_t jointCount{0};
		std::uint32_t clipTableOffset{0};  // into the shared ModelClipTableBuffer, in ModelClipEntry units
		std::uint32_t clipCount{0};
		std::uint32_t restPaletteBase{0};  // into the shared ModelClipPaletteBuffer -- this model's rest pose, jointCount matrices
		// Stage 4: into the shared ModelSkeletonBuffer, in ModelSkeletonJoint units; jointCount
		// above doubles as this range's length. kModelNoSkeletonSentinel (not 0 -- 0 is a valid
		// real base for whichever model registers first) means this model has no skeleton data at
		// all (e.g. GltfModelManager's cow producer never fills ModelAsset::skeleton) -- guards a
		// live-capable instance of a non-skeletal model from reading another model's skeleton
		// range or an uninitialized one. Was _pad0; repurposed rather than appended since this is
		// a genuinely new field occupying reserved padding, not a mid-struct insertion that would
		// shift _pad1/_pad2's offsets.
		std::uint32_t skeletonBase{0};
		std::uint32_t _pad1{0};
		std::uint32_t _pad2{0};
	};

	static_assert(sizeof(ModelDescEntry) == 64, "ModelDescEntry must match model_instance.glsl's 64-byte stride");

	inline constexpr std::uint32_t kModelNoSkeletonSentinel = 0xFFFFFFFFu;

	// Stage 4 (model/ModelSkeletonAsset, ModelLiveEvalNode): one joint's rest pose + live-eval
	// parameters, in topological order (parentIndex < this joint's own index within the model's
	// ModelSkeletonBuffer range) so ModelLiveEvalNode can compose globals with a single forward
	// serial walk. Mirrors shaders/model_instance.glsl's ModelSkeletonJoint exactly.
	struct ModelSkeletonJoint {
		glm::mat4     inverseBind{1.0f};
		glm::vec3     restTranslation{0.0f};
		float         restScale{1.0f};
		glm::vec4     restRotation{0.0f, 0.0f, 0.0f, 1.0f}; // quat xyzw
		glm::vec3     restUpAxis{0.0f, 1.0f, 0.0f};          // this joint's own "up" in its rest-local frame
		float         rigidity{1.0f};                        // 0 = fully flexible, 1 = rigid (no wind/seed bend)
		std::uint32_t parentIndex{0xFFFFFFFFu};               // 0xFFFFFFFFu = root (no parent)
		std::uint32_t _pad0{0};
		std::uint32_t _pad1{0};
		std::uint32_t _pad2{0};
	};

	static_assert(sizeof(ModelSkeletonJoint) == 128, "ModelSkeletonJoint must match model_instance.glsl's 128-byte stride");

	// Stage 4: pool sizing for ModelLiveEvalNode's transient ModelLivePaletteBuffer -- a fixed
	// budget (not instance-count-derived, since the budget IS the cap ModelCullNode's atomic
	// claim enforces) times a fixed max joint count per live-capable model (the procedural tree
	// has 3; this leaves headroom for a richer future skeleton without a buffer resize).
	inline constexpr std::uint32_t kModelLiveBudget = 256u;
	inline constexpr std::uint32_t kModelLiveMaxJointCount = 32u;

	// Stage 3: one already-culled, already-pose-resolved survivor, written by ModelCullNode
	// (model_cull.comp) and consumed by model_render.task. Carries only what the task shader
	// can't re-derive cheaply (the resolved palette bases/blend, which needed ResolveModelPose's
	// clip-table lookup) -- position/scale/rotation/modelIndex are still fetched from
	// ModelInstanceBuffer[instanceIndex] rather than duplicated here. Mirrors
	// shaders/model_instance.glsl's ModelVisibleEntry exactly.
	// Stage 4: mode is 0 for baked (paletteBaseA/B index ModelClipPaletteBuffer, as Stage 2 always
	// did) or 1 for live (paletteBaseA == paletteBaseB == the claimed live-pool slot's base,
	// blend == 0, indexing ModelLivePaletteBuffer instead -- "one skinning interface for every
	// mode," the plan's own phrase: the mesh shader's blend math never changes, only which
	// bindless buffer index payload.paletteBufIdx resolves to, chosen by model_render.task from
	// this one bit).
	struct ModelVisibleEntry {
		std::uint32_t instanceIndex{0};
		std::uint32_t paletteBaseA{0};
		std::uint32_t paletteBaseB{0};
		float         blend{0.0f};
		std::uint32_t mode{0};
	};

	static_assert(sizeof(ModelVisibleEntry) == 20, "ModelVisibleEntry must match model_instance.glsl's 20-byte stride");

	inline constexpr std::uint32_t kModelVisibleModeBaked = 0u;
	inline constexpr std::uint32_t kModelVisibleModeLive = 1u;

} // namespace brassica
