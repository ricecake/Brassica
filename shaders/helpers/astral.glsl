#ifndef HELPERS_ASTRAL_GLSL
#define HELPERS_ASTRAL_GLSL

#include "noise.glsl"

float hash13(in vec3 pos) {
	pos = fract(pos * vec3(.1031, .1030, .0973));
	pos += dot(pos, pos.zyx + 31.32);
	return fract((pos.x + pos.y) * pos.z);
}

float fbm_astral(vec3 p) {
	float v = 0.0;
	float a = 0.5;
	for (int i = 0; i < 4; i++) {
		v += a * snoise3d(p);
		p *= 2.0;
		a *= 0.5;
	}
	return v;
}

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
	return a + b * cos(6.28318530718 * (c * t + d));
}

vec3 computeNebula(vec3 dir, float time) {
	vec3 p = dir * 4.0;
	vec3 warp_offset = vec3(fbm_astral(p + time * 0.05));
	float raw_noise = fbm_astral(p + warp_offset * 0.5);

	float nebula_threshold = 0.35;
	float nebula_noise = smoothstep(nebula_threshold, 1.0, raw_noise);
	if (nebula_noise <= 0.0001) {
		return vec3(0.0);
	}

	vec3 nebula_color = palette(nebula_noise, vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.0, 0.33, 0.67));
	return 0.0003 * snoise3d(warp_offset) * nebula_color * nebula_noise;
}

vec3 computeStars(vec3 dir, float time) {
	dir = normalize(dir);

	vec3  warp = vec3(2.0, 2.0, 2.0);
	float scale = 128.0;
	vec3  id = floor(dir * scale);
	vec3  local_uv = fract(dir * scale);

	vec3 star_pos = hash33(id);
	vec3 center = vec3(0.5) + (star_pos - 0.5) * 0.5;

	vec3  global_star_dir = normalize((id + center) / scale);
	float starDensity = snoise3d(global_star_dir * warp);
	float starMask = smoothstep(-1.0, 1.50, starDensity);

	float checkHash = hash13(id);
	float starExists = step(checkHash, starMask);

	float dist = length(local_uv - center);
	float brightness = (10.0 + sin(checkHash + time / (1.0 + max(0.01, dir.y)))) * starExists;
	float radius = 0.0125 * brightness;

	float visualGlow = snoise3d(dir * warp);
	float starIntensity = 1.0 - smoothstep(radius * 0.5, radius, dist);

	vec3 starColor1 = palette(hash13(id + dir), vec3(0.17, 0.47, 0.92), vec3(0.55, 0.4, 0.4), vec3(1.0, 1.7, 1.0), vec3(0.5, 0.35, 1.0));
	vec3 starColor2 = palette(hash13(id + dir), vec3(0.48, 0.47, 0.89), vec3(0.14, 0.0, 0.63), vec3(1.5, 0.0, 0.6), vec3(0.0, 0.0, 0.0));

	vec3 backgroundGlow = 0.0001 * pow(smoothstep(-0.40, 0.9, visualGlow), 2.0) * starColor2;
	vec3 pointStars = 0.002 * starIntensity * mix(starColor1, starColor2, checkHash);

	return backgroundGlow + pointStars;
}

#endif // HELPERS_ASTRAL_GLSL
