#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
	vec4 clip = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
	vec4 view_ray = uInvProjMatrix * clip;
	vec3 world_ray = (uInvViewMatrix * vec4(view_ray.xy, -1.0, 0.0)).xyz;
	world_ray = normalize(world_ray);


	if (world_ray.y < 0.0) {
		outColor = vec4(0.4, 0.2, 0.1, 1.0f);
	}
	else {
		outColor = mix(vec4(0.1, 0.25, 1.0, 1.0f), vec4(0.0, 0.0, 0.5, 1.0f), world_ray.y);
	}
}
