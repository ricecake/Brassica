#include "helpers/constants.glsl"
#include "helpers/math.glsl"

float dot_noise(vec3 p, float phase) {
	vec3 rotated_p1 = GOLD * p;
	vec3 rotated_p2 = PHI * p * GOLD;

	// Offset the components differently so they animate out of phase
	vec3 cos_phase = rotated_p1 + vec3(phase, phase * 1.3, phase * 1.7);
	vec3 sin_phase = rotated_p2 + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

	return dot(cos(cos_phase), sin(sin_phase));
}

float dot_noise_fbm(vec3 p, float oct, float phase) {
	float val = 0.0;
	float amp = 1.0;
	float freq = 1.0;
	float max_amp = 0.0;
	for (int i = 0; i < max(0, oct); i++) {
		val += amp * dot_noise((p + (val / freq)) * freq, phase * freq);
		max_amp += amp;
		// val += amp * dot_noise(p * freq);
		amp *= 0.5;
		freq *= 2.0;
	}
	return val / max_amp;
}

vec3 cross_noise(vec3 p, float phase) {
	vec3 rotated_p1 = GOLD * p;
	vec3 rotated_p2 = PHI * p * GOLD;

	// Offset the components differently so they animate out of phase
	vec3 cos_phase = rotated_p1 + vec3(phase, phase * 1.3, phase * 1.7);
	vec3 sin_phase = rotated_p2 + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

	return cross(cos(cos_phase), sin(sin_phase));
}

vec3 cross_noise_fbm(vec3 p, float oct, float phase) {
	vec3  val = vec3(0.0);
	float amp = 1.0;
	float freq = 1.0;
	float max_amp = 0.0;
	for (int i = 0; i < max(0, oct); i++) {
		val += amp * cross_noise(p * freq + val * freq, phase * freq);
		//    val += amp * cross_noise(p*freq + val/freq, phase*freq);
		max_amp += amp;
		amp *= 0.5;
		freq *= 2.0;
	}
	return val / max_amp;
}

float dot_noise(vec3 p, float phase, out vec3 grad) {
    vec3 u = GOLD * p + vec3(phase, phase * 1.3, phase * 1.7);
    vec3 v = PHI * p * GOLD + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

    vec3 sin_u = sin(u);
    vec3 cos_u = cos(u);
    vec3 sin_v = sin(v);
    vec3 cos_v = cos(v);

    // Apply the analytical derivative
    grad = -(sin_u * sin_v) * GOLD + PHI * (GOLD * (cos_u * cos_v));

    return dot(cos_u, sin_v);
}

float dot_noise_fbm(vec3 p, int oct, float phase, out vec3 out_grad) {
    float val = 0.0;
    vec3 grad = vec3(0.0);

    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;

    for (int i = 0; i < max(0, oct); i++) {
        vec3 g_noise;

        // (p + (val/freq)) * freq simplifies algebraically to (p * freq + val)
        // Since val is a float, GLSL adds it to each component of the vec3.
        vec3 p_warp = p * freq + val;

        float n = dot_noise(p_warp, phase * freq, g_noise);

        // Accumulate the gradient using the chain rule for the domain warping.
        // The Jacobian of (p * freq + val) introduces the dot product here
        // because the scalar `val` is added uniformly across all three dimensions.
        grad += amp * (freq * g_noise + dot(g_noise, vec3(1.0)) * grad);

        val += amp * n;
        max_amp += amp;

        amp *= 0.5;
        freq *= 2.0;
    }

    out_grad = grad / max_amp;
    return val / max_amp;
}

vec3 cross_noise(vec3 p, float phase, out mat3 jacobian) {
    vec3 u = GOLD * p + vec3(phase, phase * 1.3, phase * 1.7);
    vec3 v = PHI * p * GOLD + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

    vec3 sin_u = sin(u);
    vec3 cos_u = cos(u);
    vec3 sin_v = sin(v);
    vec3 cos_v = cos(v);

    // Compute the Jacobians of the inner transformations
    mat3 J_A = -diag(sin_u) * GOLD;
    mat3 J_B =  diag(cos_v) * (PHI * transpose(GOLD));

    // Apply the cross product product-rule
    jacobian = -skew(sin_v) * J_A + skew(cos_u) * J_B;

    return cross(cos_u, sin_v);
}

vec3 cross_noise_fbm(vec3 p, int oct, float phase, out mat3 out_jacobian) {
    vec3 val = vec3(0.0);
    mat3 jacobian = mat3(0.0);

    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;

    // Identity matrix in GLSL
    mat3 I = mat3(1.0);

    for (int i = 0; i < max(0, oct); i++) {
        mat3 J_noise;

        vec3 p_warp = p * freq + val;
        vec3 n = cross_noise(p_warp, phase * freq, J_noise);

        // J_warp is the derivative of (p * freq + val)
        mat3 J_warp = freq * I + jacobian;

        // Chain rule applies standard matrix multiplication: J_outer * J_inner
        jacobian += amp * (J_noise * J_warp);

        val += amp * n;
        max_amp += amp;

        amp *= 0.5;
        freq *= 2.0;
    }

    out_jacobian = jacobian / max_amp;
    return val / max_amp;
}

float dot_noise(vec3 p, float phase, out vec3 grad, out mat3 hessian) {
    vec3 u = GOLD * p + vec3(phase, phase * 1.3, phase * 1.7);
    vec3 v = PHI * p * GOLD + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

    vec3 sin_u = sin(u); vec3 cos_u = cos(u);
    vec3 sin_v = sin(v); vec3 cos_v = cos(v);

    // M1 = GOLD
    mat3 M1 = GOLD;
    // M2 = PHI * transpose(GOLD)
    mat3 M2 = PHI * transpose(GOLD);

    // 1. First Derivative (Gradient)
    // Refactored slightly to explicit matrix multiplication for consistency with the Hessian
    grad = transpose(M1) * (-sin_u * sin_v) + transpose(M2) * (cos_u * cos_v);

    // 2. Second Derivative (Hessian)
    vec3 C1 = -cos_u * sin_v;
    vec3 C2 = -sin_u * cos_v;

    // diag() is the helper function you already defined
    mat3 grad_A = diag(C1) * M1 + diag(C2) * M2;
    mat3 grad_B = diag(C2) * M1 + diag(C1) * M2;

    hessian = transpose(M1) * grad_A + transpose(M2) * grad_B;

    return dot(cos_u, sin_v);
}

float dot_noise_fbm(vec3 p, int oct, float phase, out vec3 out_grad, out mat3 out_hessian) {
    float val = 0.0;
    vec3 grad = vec3(0.0);
    mat3 hessian = mat3(0.0);

    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;

    for (int i = 0; i < max(0, oct); i++) {
        vec3 g_noise;
        mat3 h_noise;

        vec3 p_warp = p * freq + val;
        float n = dot_noise(p_warp, phase * freq, g_noise, h_noise);

        // Jacobian of the domain warp: (p * freq + val)
        // outerProduct(vec3(1.0), grad) creates a 3x3 matrix where each row is the accumulated gradient
        mat3 J_warp = mat3(freq) + outerProduct(vec3(1.0), grad);

        // Accumulate Hessian via Chain Rule
        mat3 H_chain = transpose(J_warp) * h_noise * J_warp;

        // The term dot(g_noise, vec3(1.0)) * hessian accounts for the second derivative of the warp path
        hessian += amp * (H_chain + dot(g_noise, vec3(1.0)) * hessian);

        // Accumulate Gradient
        grad += amp * (freq * g_noise + dot(g_noise, vec3(1.0)) * grad);

        val += amp * n;
        max_amp += amp;

        amp *= 0.5;
        freq *= 2.0;
    }

    out_grad = grad / max_amp;
    out_hessian = hessian / max_amp;
    return val / max_amp;
}
