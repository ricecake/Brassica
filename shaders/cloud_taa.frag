#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outHistoryColor;   // rgb = filtered cloud radiance, a = filtered cloud transmittance
layout(location = 1) out vec4 outHistoryMoments; // temporal moments (r = E[I], g = E[I^2], b = Var(I), a = accumFrames)
layout(location = 2) out vec4 outCompositedHdr;  // final composited HDR color

layout(push_constant) uniform CloudTaaPushConstants {
	uint  cloudRenderIndex;
	uint  cloudDepthIndex;
	uint  cloudVelocityIndex;
	uint  historyColorIndex;
	uint  historyMomentsIndex;
	uint  hdrColorIndex;
	uint  frameIndex;
	float blendAlpha;
} push;

void main() {
	vec4 currCloud = SAMPLE_LINEAR(push.cloudRenderIndex, inUV);
	float currDepth = SAMPLE_LINEAR(push.cloudDepthIndex, inUV).r;
	vec2 velocity = SAMPLE_LINEAR(push.cloudVelocityIndex, inUV).rg;
	vec3 currHdr = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;

	vec2 prevUV = inUV - velocity;

	// Check if reprojected UV falls within screen bounds and if past history exists
	bool validHistory = (push.frameIndex > 0) &&
	                    (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0);

	vec4 historyColor = currCloud;
	vec4 historyMoments = vec4(currCloud.rgb, dot(currCloud.rgb, currCloud.rgb));

	if (validHistory) {
		historyColor = SAMPLE_LINEAR(push.historyColorIndex, prevUV);
		historyMoments = SAMPLE_LINEAR(push.historyMomentsIndex, prevUV);
	}

	// 3x3 Neighborhood color clamping on intermediate render for anti-ghosting
	ivec2 texSize = textureSize(sampler2D(uTextures2D[nonuniformEXT(push.cloudRenderIndex)], uSamplers[BRASSICA_SAMPLER_LINEAR_CLAMP]), 0);
	vec2 texelSize = 1.0 / vec2(texSize);

	vec4 minCloud = currCloud;
	vec4 maxCloud = currCloud;
	vec3 m1 = vec3(0.0);
	vec3 m2 = vec3(0.0);

	for (int y = -1; y <= 1; ++y) {
		for (int x = -1; x <= 1; ++x) {
			vec2 sampleUV = inUV + vec2(x, y) * texelSize;
			vec4 sampleColor = SAMPLE_LINEAR(push.cloudRenderIndex, sampleUV);
			minCloud = min(minCloud, sampleColor);
			maxCloud = max(maxCloud, sampleColor);
			m1 += sampleColor.rgb;
			m2 += sampleColor.rgb * sampleColor.rgb;
		}
	}

	m1 /= 9.0;
	m2 /= 9.0;
	vec3 sigma = sqrt(max(vec3(0.0), m2 - m1 * m1));

	// Clamp history color to 3x3 mean/stddev box
	float gamma = 1.5;
	vec3 clampMin = m1 - gamma * sigma;
	vec3 clampMax = m1 + gamma * sigma;

	historyColor.rgb = clamp(historyColor.rgb, clampMin, clampMax);
	historyColor.a = clamp(historyColor.a, minCloud.a, maxCloud.a);

	// Estimate variance and adjust accumulation weight for 16-frame subpixel accumulation
	float lumaCurr = dot(currCloud.rgb, vec3(0.2126, 0.7152, 0.0722));
	float lumaHist = dot(historyColor.rgb, vec3(0.2126, 0.7152, 0.0722));
	float variance = abs(lumaCurr - lumaHist);

	// Target ~1/16 accumulation rate (0.0625) over 16 frames when stable
	float targetAlpha = (push.blendAlpha > 0.0) ? push.blendAlpha : 0.0625;
	float alpha = validHistory ? mix(targetAlpha, 0.5, smoothstep(0.05, 0.3, variance)) : 1.0;

	vec4 blendedCloud = mix(historyColor, currCloud, alpha);

	// Update temporal moments
	float firstMoment = mix(historyMoments.r, lumaCurr, 0.0625);
	float secondMoment = mix(historyMoments.g, lumaCurr * lumaCurr, 0.0625);
	float temporalVariance = max(0.0, secondMoment - firstMoment * firstMoment);

	outHistoryColor = blendedCloud;
	outHistoryMoments = vec4(firstMoment, secondMoment, temporalVariance, validHistory ? (historyMoments.a + 1.0) : 1.0);

	// Composite anti-aliased cloud onto HDR color
	vec3 compositedRgb = currHdr * blendedCloud.a + blendedCloud.rgb;
	outCompositedHdr = vec4(compositedRgb, 1.0);
}
