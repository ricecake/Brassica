// Skew-symmetric matrix helper (takes column-major order)
mat3 skew(vec3 v) {
    return mat3(
        0.0,  v.z, -v.y,
       -v.z,  0.0,  v.x,
        v.y, -v.x,  0.0
    );
}

// Diagonal matrix helper
mat3 diag(vec3 v) {
    return mat3(
        v.x, 0.0, 0.0,
        0.0, v.y, 0.0,
        0.0, 0.0, v.z
    );
}

float cross2d(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

vec3 get_curl(mat3 J) {
    // Curl = ( dFz/dy - dFy/dz,  dFx/dz - dFz/dx,  dFy/dx - dFx/dy )
    // In GLSL: J[col][row] -> J[1][2] is dFz/dy
    return vec3(
        J[1][2] - J[2][1],
        J[2][0] - J[0][2],
        J[0][1] - J[1][0]
    );
}

float safeDiv(float a, float b) {
	return (b != 0.0) ? (a / b) : 0.0;
}

float clamp01(float t) {
	return clamp(t, 0.0, 1.0);
}

float pow_inv(float t, float power) {
	return 1.0 - pow(1.0 - clamp01(t), power);
}

vec2 safe_normalize(vec2 n) {
	float l = length(n);
	return (abs(l) > 1e-10) ? (n / l) : n;
}

vec3 hemisphereSample(vec2 uv, vec3 normal) {
	float phi = 2.0 * PI * uv.x;
	float cosTheta = sqrt(1.0 - uv.y);
	float sinTheta = sqrt(uv.y);
	vec3  localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
	vec3  up = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
	vec3  tangent = normalize(cross(up, normal));
	vec3  bitangent = cross(normal, tangent);
	return tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;
}

float roundToEvenPlaces(float value, float places) {
	float shift = pow(10.0, places);
	return roundEven(value * shift) / shift;
}

float roundToPlaces(float value, float places) {
	float shift = pow(10.0, places);
	return round(value * shift) / shift;
}

float schlickGain(float x, float g) {
	g = clamp(g, 0.001, 0.999);
	float absDiff = abs(2.0 * x - 1.0);
	float denominator = g + absDiff * (1.0 - 2.0 * g);
	return 0.5 + ((x - 0.5) * (1.0 - g)) / denominator;
}

float schlickBias(float x, float g) {
	// Guard inputs to safe analytical ranges
	float xx = clamp(x, 0.0, 1.0);
	float gg = clamp(g, 1e-4, 1.0 - 1e-4);

	// Convert bias parameter to Schlick formulation factor
	// Schlick's fast alternative: f(x) = x / ((1/a - 2) * (1.0 - x) + 1.0)
	float k = (1.0 / gg) - 2.0;
	return xx / (k * (1.0 - xx) + 1.0);
}

