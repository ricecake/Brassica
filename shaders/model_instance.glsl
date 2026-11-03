#ifndef BRASSICA_MODEL_INSTANCE_GLSL
#define BRASSICA_MODEL_INSTANCE_GLSL

#include "bindless.glsl"

// GPU-driven model instance table (model/ModelInstanceManager.hpp, passes/ModelRenderNode.hpp).
// Mirrors include/model/ModelTypes.hpp's ModelInstance exactly -- clipIndex/startTimeMs are
// compared against FrameUBO.uTimeMs entirely on the GPU; the CPU only ever writes a slot via
// ModelInstanceWrite on spawn/clip-change/despawn, never per-frame.
const uint kModelInstanceAliveBit = 1u;
const uint kModelInstanceLoopBit = 2u;
const uint kModelInstanceLiveCapableBit = 4u;
const uint kModelClipRestPose = 0xFFFFFFFFu;
const uint kModelInstanceCapacity = 1024u; // mirrors include/model/ModelTypes.hpp
const uint kModelNoSkeletonSentinel = 0xFFFFFFFFu;
const uint kModelLiveBudget = 256u;
const uint kModelLiveMaxJointCount = 32u;
const uint kModelVisibleModeBaked = 0u;
const uint kModelVisibleModeLive = 1u;

struct ModelInstance {
	vec3  position;
	float scale;
	vec4  rotation; // quat xyzw
	uint  modelIndex;
	uint  clipIndex; // kModelClipRestPose = rest pose, no palette lookup
	uint  startTimeMs;
	uint  flags;
	uint  seed; // Stage 4: per-instance phase diversity for live wind/seed-bend
	uint  _pad0;
	uint  _pad1;
	uint  _pad2;
};

// One baked clip's metadata -- mirrors ModelClipEntry. paletteBase is in *joint* units: frame f's
// joint j lives at palette[paletteBase + f*jointCount + j]. Bounds are model-local, the union
// over every baked frame's skinned vertex positions.
struct ModelClipEntry {
	uint  paletteBase;
	uint  frameCount;
	uint  durationMs;
	// Real bake-time sampling interval (1000/sampleRateHz) -- see ModelTypes.hpp's own comment
	// on why this must not be re-derived from durationMs/(frameCount-1).
	float frameDtMs;
	vec3  boundsCenter;
	float boundsRadius;
};

// One pending sparse write, scattered into ModelInstanceBuffer[slot] by
// model_instance_scatter.comp.
struct ModelInstanceWrite {
	uint          slot;
	uint          _pad0;
	uint          _pad1;
	uint          _pad2;
	ModelInstance record;
};

// One entry per registered model (model/ModelRegistry.hpp), indexed by ModelInstance::modelIndex
// -- mirrors ModelDescEntry. The *Base fields are where this model's own chunk starts within
// each shared geometry/palette/clip-table buffer (see that struct's own C++ comment for why
// rebasing happens here, at the lookup point, rather than in the data itself).
struct ModelDescEntry {
	vec3  restBoundsCenter;
	float restBoundsRadius;
	uint  vertexBase;
	uint  meshletVertexBase;
	uint  meshletTriangleWordBase;
	uint  meshletBase;
	uint  meshletCount;
	uint  jointCount;
	uint  clipTableOffset;
	uint  clipCount;
	uint  restPaletteBase;
	uint  skeletonBase; // kModelNoSkeletonSentinel = this model has no skeleton (Stage 4)
	uint  _pad1;
	uint  _pad2;
};

// Stage 4: one joint's rest pose + live-eval parameters, in topological order (parentIndex <
// this joint's own index within the model's own ModelSkeletonBuffer range) -- mirrors
// ModelSkeletonJoint.
struct ModelSkeletonJoint {
	mat4  inverseBind;
	vec3  restTranslation;
	float restScale;
	vec4  restRotation; // quat xyzw
	vec3  restUpAxis;
	float rigidity; // 0 = fully flexible, 1 = rigid (no wind/seed bend)
	uint  parentIndex; // kModelNoSkeletonSentinel = root (no parent)
	uint  _pad0;
	uint  _pad1;
	uint  _pad2;
};

// Binding 6: the generic bindless storage-buffer catalog (see material_system.glsl's own comment
// on this binding and PhysicalRegistry::AssignAndWriteBindlessBufferIndex) -- every buffer here
// is eStorageBuffer-usage and so already auto-registered; ctx.BufferIndex<K>() just reads the
// slot it was assigned, no device-address plumbing needed on the C++ side at all.
layout(set = 1, binding = 6, std430) buffer ModelInstanceBufferArray {
	ModelInstance instances[];
}
uModelInstanceBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer ModelInstanceWriteBufferArray {
	ModelInstanceWrite writes[];
}
uModelInstanceWriteBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer ModelClipTableBufferArray {
	ModelClipEntry clips[];
}
uModelClipTableBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer ModelClipPaletteBufferArray {
	mat4 matrices[];
}
uModelClipPaletteBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer ModelDescBufferArray {
	ModelDescEntry entries[];
}
uModelDescBuffers[];

layout(set = 1, binding = 6, std430) readonly buffer ModelSkeletonBufferArray {
	ModelSkeletonJoint joints[];
}
uModelSkeletonBuffers[];

// Stage 3/4: one already-culled, already-pose-resolved survivor -- mirrors ModelVisibleEntry.
// Written by model_cull.comp, read by model_render.task. mode selects which palette buffer
// paletteBaseA/B index into (kModelVisibleModeBaked -> ModelClipPaletteBuffer,
// kModelVisibleModeLive -> ModelLivePaletteBuffer) -- see that struct's own C++ comment.
struct ModelVisibleEntry {
	uint  instanceIndex;
	uint  paletteBaseA;
	uint  paletteBaseB;
	float blend;
	uint  mode;
};

layout(set = 1, binding = 6, std430) buffer ModelVisibleListBufferArray {
	ModelVisibleEntry entries[];
}
uModelVisibleListBuffers[];

// VkDrawMeshTasksIndirectCommandEXT-shaped. groupCountX doubles as model_cull.comp's live
// compaction counter, same "atomic serves double duty" idiom as FoliageIndirectArgsBuffer.
layout(set = 1, binding = 6, std430) buffer ModelIndirectArgsBufferArray {
	uint groupCountX;
	uint groupCountY;
	uint groupCountZ;
}
uModelIndirectArgsBuffers[];

// Stage 4: model_cull.comp's live-pool claim list (flat instanceIndex[], one per claimed
// slot -- the slot index IS the claim order) and its paired indirect-dispatch args for
// model_live_eval.comp. Unlike ModelIndirectArgsBuffer's one-workgroup-per-entry draw,
// model_live_eval.comp is a plain many-threads-per-workgroup compute dispatch, so groupCountX
// here is ceil(rawLiveCount/64) -- a real divide ModelLiveFinalizeNode performs between
// model_cull.comp's atomicAdd into rawLiveCount and this dispatch, not rawLiveCount itself.
layout(set = 1, binding = 6, std430) buffer ModelLiveListBufferArray {
	uint instanceIndex[];
}
uModelLiveListBuffers[];

layout(set = 1, binding = 6, std430) buffer ModelLiveIndirectArgsBufferArray {
	uint groupCountX;
	uint groupCountY;
	uint groupCountZ;
	uint rawLiveCount;
}
uModelLiveIndirectArgsBuffers[];

// Flat mat4[kModelLiveBudget * kModelLiveMaxJointCount], addressed as
// liveSlot * kModelLiveMaxJointCount + joint.
layout(set = 1, binding = 6, std430) buffer ModelLivePaletteBufferArray {
	mat4 matrices[];
}
uModelLivePaletteBuffers[];

vec3 RotateByQuat(vec4 q, vec3 v) {
	return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Rotates v by q's inverse (q is always unit-length here, so inverse == conjugate) -- used by
// model_live_eval.comp to convert a world-space wind direction into an instance's own local
// space once per instance (the plan's own phrase), rather than per joint.
vec3 InverseRotateByQuat(vec4 q, vec3 v) {
	return RotateByQuat(vec4(-q.xyz, q.w), v);
}

// Hamilton product, a then b applied (R(a)*R(b)*v applies b first) -- model_live_eval.comp's
// "rest * seedBend * windBend" composition (the plan's own formula) is this called twice,
// left-to-right exactly as written.
vec4 QuatMul(vec4 a, vec4 b) {
	return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// Builds T*R*S as one mat4 directly (translation in the last column, R*S in the 3x3 block) --
// equivalent to translate(t) * mat4_cast(q) * scale(vec3(s)), matching GltfModel::EvaluatePose/
// ProceduralTreeAsset's own TRS composition convention exactly.
mat4 QuatTranslationScaleToMat4(vec3 t, vec4 q, float s) {
	float x = q.x, y = q.y, z = q.z, w = q.w;
	mat3 r = mat3(
		1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
		2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
		2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)
	) * s;
	return mat4(vec4(r[0], 0.0), vec4(r[1], 0.0), vec4(r[2], 0.0), vec4(t, 1.0));
}

// Resolves an instance's (paletteA, paletteB, blend) against the rest-pose/clip-table rules
// described above -- the one place both baked skinning (model_render.mesh) and any future live-
// mode fallback need to agree on this logic, so there is exactly one skinning code path
// regardless of mode (see the plan's "one skinning interface for every mode"). desc.jointCount/
// clipTableOffset/restPaletteBase carry what used to be single-model push-constant values --
// see ModelDescEntry's own comment.
void ResolveModelPose(
	ModelInstance  inst,
	ModelDescEntry desc,
	uint           timeMs,
	uint           clipTableBufIdx,
	out uint       paletteBaseA,
	out uint       paletteBaseB,
	out float      blend
) {
	// >= clipCount (not just == kModelClipRestPose) is deliberate: a corrupted/stale clipIndex
	// from a previous model falls back to this model's own rest pose instead of reading past
	// the end of its clip-table range into some other model's clips.
	if (inst.clipIndex >= desc.clipCount) {
		paletteBaseA = desc.restPaletteBase;
		paletteBaseB = desc.restPaletteBase;
		blend = 0.0f;
		return;
	}

	ModelClipEntry clip = uModelClipTableBuffers[nonuniformEXT(clipTableBufIdx)].clips[desc.clipTableOffset + inst.clipIndex];
	// clip.paletteBase is model-local (relative to this model's own palette chunk, which starts
	// with its rest pose -- see GltfModel::BakeClips/ProceduralTreeAsset, neither of which knows
	// about any other model) -- desc.restPaletteBase is where that chunk starts in the *shared*
	// palette buffer, so every paletteBase below needs it added to become a real global offset.
	if (clip.frameCount <= 1u || clip.durationMs == 0u) {
		paletteBaseA = desc.restPaletteBase + clip.paletteBase;
		paletteBaseB = desc.restPaletteBase + clip.paletteBase;
		blend = 0.0f;
		return;
	}

	// Signed on purpose: a startTimeMs in the future (not started yet) must hold frame 0, not
	// wrap around to a huge unsigned elapsed time. Exact for any realistic spawn-to-now gap --
	// signed int32 covers +-24.8 days, and a slot's startTimeMs is always set near "now" at
	// spawn, never far in the past/future -- while still correctly handling uTimeMs's own ~49.7
	// day wrap, since both sides wrap together and the signed difference stays small either way.
	int  elapsedMsSigned = int(timeMs) - int(inst.startTimeMs);
	uint elapsedMs = elapsedMsSigned > 0 ? uint(elapsedMsSigned) : 0u;
	if ((inst.flags & kModelInstanceLoopBit) != 0u) {
		elapsedMs = elapsedMs % clip.durationMs;
	} else {
		elapsedMs = min(elapsedMs, clip.durationMs);
	}

	float frameF = float(elapsedMs) / clip.frameDtMs;
	uint  f0 = min(uint(frameF), clip.frameCount - 2u);
	blend = frameF - float(f0);
	paletteBaseA = desc.restPaletteBase + clip.paletteBase + f0 * desc.jointCount;
	paletteBaseB = desc.restPaletteBase + clip.paletteBase + (f0 + 1u) * desc.jointCount;
}

#endif // BRASSICA_MODEL_INSTANCE_GLSL
