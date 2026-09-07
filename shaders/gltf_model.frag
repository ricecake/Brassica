#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct GltfMaterialGPU {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureIndex;
    int normalTextureIndex;
    int metallicRoughnessTextureIndex;
    int pad0, pad1, pad2;
};

layout(std430, set = 1, binding = 5) readonly buffer MaterialBuffer {
    GltfMaterialGPU materials[];
};

// Set 2: Bindless Texture Array
layout(set = 2, binding = 0) uniform sampler2D u_Textures[];

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) flat in uint inMaterialIndex;

layout(location = 0) out vec4 outGBufferPos;
layout(location = 1) out vec4 outGBufferNormal;
layout(location = 2) out vec4 outGBufferAlbedo;

void main() {
    GltfMaterialGPU mat = materials[inMaterialIndex];

    vec4 albedo = mat.baseColorFactor;
    if (mat.baseColorTextureIndex >= 0) {
        vec4 texColor = texture(u_Textures[nonuniformEXT(mat.baseColorTextureIndex)], inUV);
        albedo *= texColor;
    }

    vec3 normal = normalize(inNormal);
    if (mat.normalTextureIndex >= 0) {
        vec3 mapNorm = texture(u_Textures[nonuniformEXT(mat.normalTextureIndex)], inUV).rgb * 2.0 - 1.0;
        // Basic tangent space approximation
        normal = normalize(normal + mapNorm * 0.5);
    }

    outGBufferPos = vec4(inWorldPos, 1.0);
    outGBufferNormal = vec4(normal, 0.0);
    outGBufferAlbedo = albedo;
}
