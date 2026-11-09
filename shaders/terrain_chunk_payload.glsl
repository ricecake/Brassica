#ifndef BRASSICA_TERRAIN_CHUNK_PAYLOAD_GLSL
#define BRASSICA_TERRAIN_CHUNK_PAYLOAD_GLSL

struct TerrainChunkTaskPayload {
	uint physicalSlots[32];
	uint virtualChunkIds[32];
	uint lodLevels[32];
};

taskPayloadSharedEXT TerrainChunkTaskPayload chunkPayload;

#endif // BRASSICA_TERRAIN_CHUNK_PAYLOAD_GLSL
