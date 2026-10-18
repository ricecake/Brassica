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

vec3 RGBToYCoCg(vec3 rgb) {
	float y  = dot(rgb, vec3(0.25, 0.5, 0.25));
	float co = dot(rgb, vec3(0.5, 0.0, -0.5));
	float cg = dot(rgb, vec3(-0.25, 0.5, -0.25));
	return vec3(y, co, cg);
}

vec3 YCoCgToRGB(vec3 ycocg) {
	float r = ycocg.x + ycocg.y - ycocg.z;
	float g = ycocg.x + ycocg.z;
	float b = ycocg.x - ycocg.y - ycocg.z;
	return vec3(r, g, b);
}

void main() {
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	uint f = push.frameIndex % 16u;
	ivec2 subOffset = ivec2(int(f % 4u), int(f / 4u));
	bool isFreshThisFrame = (ivec2(pixel.x % 4, pixel.y % 4) == subOffset);

	vec4 currCloud = SAMPLE_NEAREST(push.cloudRenderIndex, inUV);
	float currDepth = SAMPLE_NEAREST(push.cloudDepthIndex, inUV).r;
	vec3 currHdr = SAMPLE_NEAREST(push.hdrColorIndex, inUV).rgb;

	vec2 velocity = vec2(0.0);
	if (isFreshThisFrame && currDepth < 1e8) {
		velocity = SAMPLE_NEAREST(push.cloudVelocityIndex, inUV).rg;
	} else {
		vec2 clipCoord = inUV * 2.0 - 1.0;
		vec4 viewRay4 = uInvProjMatrix * vec4(clipCoord, 1.0, 1.0);
		vec3 viewDir = viewRay4.xyz / viewRay4.w;
		vec3 worldRay = normalize((uInvViewMatrix * vec4(viewDir, 0.0)).xyz);
		float depth = 5000.0;
		vec3 cloudWorldPos = uCameraPosition.xyz + worldRay * depth;
		vec4 prevClip = uPreviousViewProjMatrix * vec4(cloudWorldPos, 1.0);
		if (prevClip.w > 0.001) {
			vec2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
			velocity = inUV - prevUV;
		}
	}

	vec2 prevUV = inUV - velocity;

	// Check if reprojected UV falls within screen bounds and if past history exists
	bool validHistory = (push.frameIndex > 0) &&
	                    (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0);

	vec4 historyColor = validHistory ? SAMPLE_LINEAR(push.historyColorIndex, prevUV) : (isFreshThisFrame ? currCloud : vec4(0.0, 0.0, 0.0, 1.0));
	vec4 historyMoments = validHistory ? SAMPLE_LINEAR(push.historyMomentsIndex, prevUV) : vec4(currCloud.rgb, dot(currCloud.rgb, currCloud.rgb));

	vec4 blendedCloud;

	if (isFreshThisFrame) {
		ivec2 texSize = textureSize(sampler2D(uTextures2D[nonuniformEXT(push.cloudRenderIndex)], uSamplers[BRASSICA_SAMPLER_NEAREST_CLAMP]), 0);
		vec2 texelSize = 1.0 / vec2(texSize);

		vec3 m1 = vec3(0.0);
		vec3 m2 = vec3(0.0);
		float minAlpha = currCloud.a;
		float maxAlpha = currCloud.a;

		for (int y = -1; y <= 1; ++y) {
			for (int x = -1; x <= 1; ++x) {
				vec2 sampleUV = inUV + vec2(x, y) * texelSize * 4.0;
				vec4 sampleColor = SAMPLE_NEAREST(push.cloudRenderIndex, sampleUV);
				vec3 ycocg = RGBToYCoCg(sampleColor.rgb);
				m1 += ycocg;
				m2 += ycocg * ycocg;
				minAlpha = min(minAlpha, sampleColor.a);
				maxAlpha = max(maxAlpha, sampleColor.a);
			}
		}

		m1 /= 9.0;
		m2 /= 9.0;
		vec3 sigma = sqrt(max(vec3(0.0), m2 - m1 * m1));

		float gamma = 1.5;
		vec3 clampMin = m1 - gamma * sigma;
		vec3 clampMax = m1 + gamma * sigma;

		vec3 histYCoCg = RGBToYCoCg(historyColor.rgb);
		histYCoCg = clamp(histYCoCg, clampMin, clampMax);

		historyColor.rgb = YCoCgToRGB(histYCoCg);
		historyColor.a = clamp(historyColor.a, minAlpha, maxAlpha);

		float alpha = validHistory ? 0.15 : 1.0;
		blendedCloud = mix(historyColor, currCloud, alpha);
	} else {
		// Propagate reprojected history smoothly without raymarching on unrendered threads
		blendedCloud = historyColor;
	}

	// Update temporal moments
	float lumaCurr = dot(blendedCloud.rgb, vec3(0.2126, 0.7152, 0.0722));
	float firstMoment = mix(historyMoments.r, lumaCurr, 0.0625);
	float secondMoment = mix(historyMoments.g, lumaCurr * lumaCurr, 0.0625);
	float temporalVariance = max(0.0, secondMoment - firstMoment * firstMoment);

	outHistoryColor = blendedCloud;
	outHistoryMoments = vec4(firstMoment, secondMoment, temporalVariance, validHistory ? (historyMoments.a + 1.0) : 1.0);

	// Composite anti-aliased cloud onto HDR color
	vec3 compositedRgb = currHdr * blendedCloud.a + blendedCloud.rgb;
	outCompositedHdr = vec4(compositedRgb, 1.0);
}
