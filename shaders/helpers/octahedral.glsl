#ifndef OCTAHEDRAL_GLSL
#define OCTAHEDRAL_GLSL

// Octahedral projection mapping functions between 3D unit direction vectors and 2D octahedral UV coordinates [0, 1]^2

vec2 octWrap(vec2 v) {
	return (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// Maps a 3D direction vector to 2D octahedral UV coordinates in [0, 1]^2
vec2 directionToOctahedralUV(vec3 dir) {
	vec3 rawDir = vec3(dir.x, dir.z, -dir.y);
	rawDir /= (abs(rawDir.x) + abs(rawDir.y) + abs(rawDir.z) + 1e-6);
	vec2 oct = (rawDir.y >= 0.0) ? rawDir.xz : octWrap(rawDir.xz);
	return oct * 0.5 + 0.5;
}

// Maps 2D octahedral UV coordinates in [0, 1]^2 to a 3D unit direction vector on the sphere
vec3 octahedralUVToDirection(vec2 uv) {
	vec2 oct = uv * 2.0 - 1.0;
	vec3 rawDir = vec3(oct.x, 1.0 - abs(oct.x) - abs(oct.y), oct.y);
	if (rawDir.y < 0.0) {
		rawDir.xz = octWrap(rawDir.xz);
	}
	vec3 nRaw = normalize(rawDir);
	return vec3(nRaw.x, -nRaw.z, nRaw.y);
}

#endif // OCTAHEDRAL_GLSL
