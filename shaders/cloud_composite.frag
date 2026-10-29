#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"

// Blends cloud_raymarch.comp's output into HdrColor -- the other half of the Stage A split
// (compute raymarch produces CloudColor/CloudTransmittance, this trivial full-screen pass
// composites them). Same math cloud_render.frag's old main() did directly:
// currentRadiance = currentRadiance * resultTransmittance + resultRgb.
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CloudCompositePushConstants {
	uint hdrColorIndex;
	uint cloudColorIndex;
	uint cloudTransmittanceIndex;
} push;

void main() {
	vec3 currentRadiance = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;
	vec3 cloudRgb = SAMPLE_NEAREST(push.cloudColorIndex, inUV).rgb;
	vec3 cloudTransmittance = SAMPLE_NEAREST(push.cloudTransmittanceIndex, inUV).rgb;
	outColor = vec4(currentRadiance * cloudTransmittance + cloudRgb, 1.0);
}
