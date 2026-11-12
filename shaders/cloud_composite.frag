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
	uint gPositionIndex;
	uint gAlbedoIndex;
	uint cloudMomentsIndex;
} push;

void main() {
	vec3 currentRadiance = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;
	vec3 cloudRgb = SAMPLE_NEAREST(push.cloudColorIndex, inUV).rgb;
	vec3 cloudTransmittance = SAMPLE_NEAREST(push.cloudTransmittanceIndex, inUV).rgb;

	// Full-resolution safety net: cloud_temporal.comp operates on a coarse, 4x4-block-granularity
	// grid, so a thin or dithered foreground object's exact silhouette pixel can still end up
	// reusing a neighboring sample's accumulated cloud. CloudMoments.w holds the accumulated
	// cloud's own representative depth (1e9 if there's no visible cloud there) -- if this exact
	// pixel has a real surface strictly nearer than that, the surface occludes it and the cloud
	// must not show here, regardless of what the coarser upstream grid decided.
	vec4  albedo = SAMPLE_NEAREST(push.gAlbedoIndex, inUV);
	vec3  relPos = SAMPLE_NEAREST(push.gPositionIndex, inUV).rgb;
	float cloudDepth = SAMPLE_NEAREST(push.cloudMomentsIndex, inUV).w;
	if (albedo.a >= 0.01 && length(relPos) < cloudDepth) {
		cloudTransmittance = vec3(1.0);
		cloudRgb = vec3(0.0);
	}

	// No meaningful cloud contribution here -- discard rather than write. The color result would
	// be currentRadiance * 1.0 + 0.0, i.e. HdrColor unchanged, so discarding (leaving the loaded
	// value in place) is visually identical to writing it back, and it's what skips this pass's
	// own stencil write (GraphicsPipelineState::stencilMode = Write, reference = 2, set via
	// CloudCompositeNode's BindGraphics call) for a pixel that isn't actually cloud -- a discarded
	// fragment never reaches the late per-fragment stencil-write stage. Threshold rather than an
	// exact ==1.0/==0.0 check since the raymarch/temporal accumulation can leave tiny non-zero
	// residue that isn't visually "cloud" (this also covers the occluded-by-nearer-surface case
	// just above, which resets to exactly this transmittance/color).
	float avgTransmittance = (cloudTransmittance.r + cloudTransmittance.g + cloudTransmittance.b) / 3.0;
	float cloudRgbMax = max(cloudRgb.r, max(cloudRgb.g, cloudRgb.b));
	if ((1.0 - avgTransmittance) < 0.01 && cloudRgbMax < 0.001) {
		discard;
	}

	outColor = vec4(currentRadiance * cloudTransmittance + cloudRgb, 1.0);
}
