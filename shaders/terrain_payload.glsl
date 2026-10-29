#ifndef BRASSICA_TERRAIN_PAYLOAD_GLSL
#define BRASSICA_TERRAIN_PAYLOAD_GLSL

// Shared between terrain.task (the sole writer) and terrain.mesh (the sole reader) -- before this
// file existed, both shaders carried their own hand-copied TaskPayload definition, and nothing
// caught them drifting apart: adding macroColRow to terrain.task's copy without also adding it
// here left terrain.mesh referencing a field that, from its own compile's point of view, simply
// didn't exist. One definition, included by both, makes that class of bug impossible instead of
// just unlikely.
struct TaskPayload {
    vec2 relMeshletMinXZ[512];
    uint renderLOD[512];
    uint morphLOD[512];
    uint childXY[512];      // (cy << 16) | cx -- local grid position within the macro meshlet
    uint macroColRow[512];  // (row << 16) | col -- camera-position-independent grid index of this
                             // meshlet's macro slot, so terrain.mesh can rebuild a clean (non-
                             // camera-round-tripped) clipmap sample anchor; see its own comment.
};

taskPayloadSharedEXT TaskPayload payload;

#endif // BRASSICA_TERRAIN_PAYLOAD_GLSL
