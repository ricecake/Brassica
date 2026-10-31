#version 460

// Dummy target only: this pipeline's real output is foliage_bake.mesh's buffer writes, not
// anything rasterized here. A graphics pipeline still needs a fragment stage and at least one
// attachment to be valid (DynamicRenderingWrapper::Begin requires a real Write-access Image2D/
// Image3D realization) -- see FoliageBakeNode.hpp's FoliageBakeDummyTarget.
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4(0.0);
}
