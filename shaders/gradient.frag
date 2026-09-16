#version 460
#include "lighting.glsl"

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main() {
	vec4 clip = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
	vec4 view_ray = uInvProjMatrix * clip;
	vec3 world_ray = (uInvViewMatrix * vec4(view_ray.xy, -1.0, 0.0)).xyz;
	world_ray = normalize(world_ray);

	vec3 skyColor;
	if (world_ray.y < 0.0) {
		skyColor = vec3(0.4, 0.2, 0.1);
	}
	else {
		skyColor = mix(vec3(0.1, 0.25, 1.0), vec3(0.0, 0.0, 0.5), world_ray.y);
	}

	uint count = min(uLightCount, 2u);
	for (uint i = 0u; i < count; ++i) {
		if (uLights[i].type == LIGHT_TYPE_DIRECTIONAL && uLights[i].intensity > 0.0) {
			vec3 lightDir = normalize(-uLights[i].direction);
			float cosAngle = dot(world_ray, lightDir);
			if (cosAngle > 0.0) {
				vec3 lightColor = uLights[i].color * uLights[i].intensity;
				float brightness = length(lightColor);
				if (brightness > 0.001) {
					float angle = acos(clamp(cosAngle, -1.0, 1.0));
					float radius = 0.02 * brightness;
					float disc = smoothstep(radius, radius * 0.7, angle);
					float halo = pow(cosAngle, max(1.0, 200.0 / brightness)) * 0.25;
					skyColor += lightColor * (disc + halo);
				}
			}
		}
	}

	outColor = vec4(skyColor, 1.0f);
}
