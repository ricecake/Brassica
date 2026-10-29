#version 460

layout(location = 0) in struct LightningVertexOutput {
	vec3 color;
} IN;

layout(location = 0) out vec4 outColor;

void main() {
	// Standard alpha blend (PipelineLibrary's enableBlend preset, same as particle.mesh's own
	// billboards) -- alpha tracks brightness so a fully-faded strike contributes nothing.
	float alpha = clamp(max(IN.color.r, max(IN.color.g, IN.color.b)), 0.0, 1.0);
	outColor = vec4(IN.color, alpha);
}
