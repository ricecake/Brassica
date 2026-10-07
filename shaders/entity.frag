#version 460

#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec4 vColor;
layout(location = 3) in vec4 vMaterial;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec4 outMaterial;

void main() {
    outPosition = vec4(vWorldPos - uCameraPosition.xyz, vMaterial.w); // w = emissivity
    outNormal = vec4(normalize(vNormal), 1.0); // a = AO
    outAlbedo = vec4(vColor.rgb, vColor.a);
    outMaterial = vec4(vMaterial.x, vMaterial.y, vMaterial.z, 0.0); // metallic, roughness, glint
}
