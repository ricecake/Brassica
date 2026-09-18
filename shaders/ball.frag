#version 460

#include "bindless.glsl"
#include "common.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec4 vColor;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outAlbedo;

void main() {
    outPosition = vec4(vWorldPos - uCameraPosition.xyz, 1.0);
    outNormal = vec4(normalize(vNormal), 0.0);
    outAlbedo = vec4(vColor.rgb, 1.0); // Bright blue material
}
