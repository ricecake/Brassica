float dot_noise(vec3 p, float phase) {
    #ifndef PHI
    #define PHI 1.618033988749894848204586834
    #endif

    const mat3 GOLD = mat3(
    -0.571464913, +0.814921382, +0.096597072,
    -0.278044873, -0.303026659, +0.911518454,
    +0.772087367, +0.494042493, +0.399753815);

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
		val += amp * dot_noise((p+(val/freq)) * freq, phase * freq);
		max_amp += amp;
		// val += amp * dot_noise(p * freq);
		amp *= 0.5;
		freq *= 2.0;
	}
	return val / max_amp;
}

vec3 cross_noise(vec3 p, float phase) {
    #ifndef PHI
    #define PHI 1.618033988749894848204586834
    #endif

    const mat3 GOLD = mat3(
    -0.571464913, +0.814921382, +0.096597072,
    -0.278044873, -0.303026659, +0.911518454,
    +0.772087367, +0.494042493, +0.399753815);

    vec3 rotated_p1 = GOLD * p;
    vec3 rotated_p2 = PHI * p * GOLD;

    // Offset the components differently so they animate out of phase
    vec3 cos_phase = rotated_p1 + vec3(phase, phase * 1.3, phase * 1.7);
    vec3 sin_phase = rotated_p2 + vec3(phase * 1.1, phase * 0.7, phase * 1.5);

    return cross(cos(cos_phase), sin(sin_phase));
}

vec3 cross_noise_fbm(vec3 p, float oct, float phase) {
       vec3 val = vec3(0.0);
       float amp = 1.0;
       float freq = 1.0;
       float max_amp = 0.0;
       for (int i = 0; i < max(0, oct); i++) {
               val += amp * cross_noise(p*freq + val*freq, phase*freq);
            //    val += amp * cross_noise(p*freq + val/freq, phase*freq);
               max_amp += amp;
               amp *= 0.5;
               freq *= 2.0;
       }
       return val / max_amp;
}

