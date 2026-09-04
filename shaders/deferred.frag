#version 460

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUBO {
	float time;
	uint  frameIndex;
	uint  globalSeed;
	uint  frameRandom;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D gPosition;
layout(set = 1, binding = 1) uniform sampler2D gNormal;
layout(set = 1, binding = 2) uniform sampler2D gAlbedo;
layout(set = 1, binding = 3) uniform sampler2D backgroundTex;

void main() {
	vec4 albedo = texture(gAlbedo, inUV);
	vec3 norm = texture(gNormal, inUV).rgb;
	vec3 pos = texture(gPosition, inUV).rgb;
	vec4 bg = texture(backgroundTex, inUV);

	// If no surface rendered in gbuffer (albedo alpha is 0), show background
	if (albedo.a < 0.01) {
		outColor = bg;
		return;
	}

	vec3 lightDir = normalize(vec3(sin(ubo.time), sin(ubo.time * 0.5) * 0.5 + 0.5, -cos(ubo.time)));
	float diff = max(dot(norm, lightDir), 0.15);

	outColor = vec4(albedo.rgb * diff, 1.0);
}
