
const float FAKE_PLANET_RADIUS = 600000.0; // 600km radius (1/10th scale planet)
const float PI = 3.14159265359;
const float PHI = 1.618033988749894848204586834;
const float TAU = 2.0 * PI;

const mat3 GOLD = mat3(
	-0.571464913,
	+0.814921382,
	+0.096597072,
	-0.278044873,
	-0.303026659,
	+0.911518454,
	+0.772087367,
	+0.494042493,
	+0.399753815
);


// const float PI = 3.14159265359;
// const float TAU = 2.0 * PI;
// const float PHI = 1.61803398875;

// const mat3 GOLD = mat3(
//     -0.571464913, +0.814921382, +0.096597072,
//     -0.278044873, -0.303026659, +0.911518454,
//     +0.772087367, +0.494042493, +0.399753815
// );


const int bayer4x4[16] = int[](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);

float safeDiv(float a, float b) {
	return (b != 0.0) ? (a / b) : 0.0;
}

float luminance(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// A standard 32-bit integer hash
uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

vec2 hash(vec2 p) {
	p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
	return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

float pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return float((word >> 22u) ^ word) / 4294967295.0;
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

ivec2 involute(ivec2 curr, uint dim, uint seed) {
	// Assuming N is a power of 2, mask = N - 1
	uint mask = dim - 1;
	uint S = uint(curr.x) + uint(curr.y);

	// Generate arbitrary non-linear noise based on the invariant sum
	uint noise = hash(seed ^ hash(S));

	// Clear the lowest bit of the noise, then insert the opposite parity of S
	uint F = (noise & ~1u) | ((S + 1u) & 1u);

	// Apply the transformation
	uint x_new = (curr.y + F) & mask;
	uint y_new = (curr.x - F) & mask;
	return ivec2(x_new, y_new);
}

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

float roundToEvenPlaces(float value, float places) {
	float shift = pow(10.0, places);
	return roundEven(value * shift) / shift;
}

float roundToPlaces(float value, float places) {
	float shift = pow(10.0, places);
	return round(value * shift) / shift;
}

float terraceSmooth(float h, float numSteps, float slopeCoarseness) {
	float stepId = floor(h * numSteps);
	float fractional = fract(h * numSteps);

	// Smooth the transition edge between steps
	// slopeCoarseness: 0.0 = perfectly sharp, 1.0 = completely smooth
	float edge = smoothstep(0.0, slopeCoarseness, fractional);

	return (stepId + edge) / numSteps;
}

// High-quality 32-bit integer hash to generate deterministic pseudo-random seeds
uint hashUint(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

// Hierarchical Base-4 Owen Scramble for a 32-bit Morton Code
uint owenScrambleBase4(uint mortonCode, uint seed) {
	uint scrambled = 0U;
	uint currentSeed = seed;

	// Process 16 pairs of bits (32 bits total for a 2D Morton code)
	// We go from the highest-significance digit to the lowest
	for (int i = 15; i >= 0; i--) {
		// Extract the current base-4 digit (2 bits)
		uint digitShift = uint(i * 2);
		uint digit = (mortonCode >> digitShift) & 3U;

		// Generate a pseudo-random 2-bit permutation based on the structural path history
		// Mixing the current path seed with a hash creates the hierarchical scrambling
		currentSeed = hashUint(currentSeed ^ (digit + uint(i)));
		uint permutation = currentSeed & 3U;

		// Apply the permutation (XOR is standard for Owen scrambling)
		uint scrambledDigit = digit ^ permutation;

		// Reconstruct the scrambled code
		scrambled |= (scrambledDigit << digitShift);

		// Feed the scrambled digit forward to downstream children to preserve hierarchy
		currentSeed ^= scrambledDigit;
	}

	return scrambled;
}

// Spreads 16 bits of a uint out to every other bit (32 bits total)
uint part1by1(uint n) {
	n &= 0x0000ffffu;                  // n = ---- ---- ---- ---- fedc ba98 7654 3210
	n = (n ^ (n << 8u)) & 0x00ff00ffu; // n = ---- ---- fedc ba98 ---- ---- 7654 3210
	n = (n ^ (n << 4u)) & 0x0f0f0f0fu; // n = ---- fedc ---- ba98 ---- 7654 ---- 3210
	n = (n ^ (n << 2u)) & 0x33333333u; // n = --fe --dc --ba --98 --76 --54 --32 --10
	n = (n ^ (n << 1u)) & 0x55555555u; // n = f e d c b a 9 8 7 6 5 4 3 2 1 0
	return n;
}

// Compacts every other bit of a 32-bit uint back into 16 contiguous bits
uint unpart1by1(uint n) {
	n &= 0x55555555u;                  // n = f e d c b a 9 8 7 6 5 4 3 2 1 0
	n = (n ^ (n >> 1u)) & 0x33333333u; // n = --fe --dc --ba --98 --76 --54 --32 --10
	n = (n ^ (n >> 2u)) & 0x0f0f0f0fu; // n = ---- fedc ---- ba98 ---- 7654 ---- 3210
	n = (n ^ (n >> 4u)) & 0x00ff00ffu; // n = ---- ---- fedc ba98 ---- ---- 7654 3210
	n = (n ^ (n >> 8u)) & 0x0000ffffu; // n = ---- ---- ---- ---- fedc ba98 7654 3210
	return n;
}

// ENCODE: Interleaves two 16-bit values into a 32-bit index
uint encodeMorton2D(uvec2 coords) {
	return part1by1(coords.x) | (part1by1(coords.y) << 1u);
}

// DECODE: Extracts two 16-bit coordinates from a 32-bit Morton code
uvec2 decodeMorton2D(uint code) {
	return uvec2(unpart1by1(code), unpart1by1(code >> 1u));
}

uint mortonOwenScramble(uvec2 p, uint seed) {
	uint morton = encodeMorton2D(p);
	return owenScrambleBase4(morton, seed);
}

float mortonOwenThreshold(ivec2 uv, int FrameId) {
	// float temporalShift = fract(float(FrameId) * 0.61803398);
	uint code = mortonOwenScramble(uvec2(uv), uint(FrameId));
	return fract(uintBitsToFloat(code)); // + temporalShift);
}

float mortonOwenThreshold(vec2 uv, int FrameId) {
	return mortonOwenThreshold(ivec2(uv * 8192), FrameId);
}

float henyeyGreenstein(float g, float cosTheta) {
	float g2 = g * g;
	return (1.0 - g2) / (4.0 * PI * pow(max(0.0001, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

float remap(float value, float low1, float high1, float low2, float high2) {
	return low2 + (value - low1) * (high2 - low2) / max(0.0001, (high1 - low1));
}

float bayer4x4StepPhase(ivec2 pixel, int index) {
	return float(bayer4x4[((pixel.x & 3) * 4 + (pixel.y & 3)) % 16]) / 16.0;
}

float InterleavedGradientNoise(vec2 uv, int FrameId) {
	// uv += float(FrameId)  * (vec2(47, 17) * 0.695f);
	// vec3 magic = vec3( 12.9898, 78.233, 43758.5453123 );
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	float      spatialJitter = fract(magic.z * fract(dot(uv, magic.xy)));
	float      temporalShift = fract(float(FrameId) * 0.61803398);
	return fract(spatialJitter + temporalShift);
}

/**
 * Simple Phacelle Noise (2D)
 * Approximates highly directional phasor noise using a 16-sample kernel.
 *
 * @param uv  - The sampling position. Scale this to change the density of the noise cells.
 * @param dir - The desired direction of the ripples. Does not need to be normalized if
 *              you want direction magnitude to influence the phase gradient.
 * @return    - A normalized vec2 representing [cos(phase), sin(phase)].
 *              Extract the .x component for the base wave pattern.
 */
vec2 fastSimplePhacelle2d(vec2 uv, vec2 dir) {
	vec2 cell = floor(uv);
	vec2 frac = fract(uv);

	float sumCos = 0.0;
	float sumSin = 0.0;
	float sumWeight = 0.0;

	// Evaluate 4x4 grid for smooth overlapping kernels
	for (int y = -1; y <= 2; y++) {
		for (int x = -1; x <= 2; x++) {
			vec2 offset = vec2(float(x), float(y));
			vec2 neighborCell = cell + offset;

			// Generate a random, static phase shift for this specific cell [0, 2PI]
			float cellPhase = fract(sin(dot(neighborCell, vec2(12.9898, 78.233))) * 43758.5453) * 6.28318530718;

			// Vector from current sampling point to the neighboring cell's origin
			vec2  delta = offset - frac;
			float distSq = dot(delta, delta);

			// Kernel falloff weight (using a fast polynomial approximation of a Gaussian)
			// Cutoff at squared distance 4.0
			float weight = max(0.0, 1.0 - distSq * 0.25);
			weight = weight * weight * weight;

			// Project the spatial delta onto the desired direction vector to align the wave,
			// then add the cell's random phase offset.
			float phase = cellPhase + dot(dir, delta) * 3.14159265;

			// Accumulate both wave components
			sumCos += cos(phase) * weight;
			sumSin += sin(phase) * weight;
			sumWeight += weight;
		}
	}

	// Normalize the accumulated vector to rebuild a clean phase
	return normalize(vec2(sumCos, sumSin) / (sumWeight + 0.0001));
}

// The Simple Phacelle Noise function produces a stripe pattern aligned with the input vector.
// The name Phacelle is a portmanteau of phase and cell, since the function produces a phase by
// interpolating cosine and sine waves from multiple cells.
//  - p is the input point being evaluated.
//  - normDir is the direction of the stripes at this point. It must be a normalized vector.
//  - freq is the freqency of the stripes within each cell. It's best to keep it close to 1.0.
//  - offset is the phase offset of the stripes, where 1.0 is a full cycle.
//  - normalization is the degree of normalization applied, between 0 and 1.
vec4 PhacelleNoise(in vec2 p, vec2 normDir, float freq, float offset, float normalization) {
	// Get a vector orthogonal to the input direction, with a
	// magnitude proportional to the frequency of the stripes.
	vec2 sideDir = normDir.yx * vec2(-1.0, 1.0) * freq * TAU;
	offset *= TAU;

	vec2  pInt = floor(p);
	vec2  pFrac = fract(p);
	vec2  phaseDir = vec2(0.0);
	float weightSum = 0.0;
	for (int i = -1; i <= 2; i++) {
		for (int j = -1; j <= 2; j++) {
			vec2 gridOffset = vec2(i, j);
			vec2 gridPoint = pInt + gridOffset;
			vec2 randomOffset = hash(gridPoint) * 0.5;
			vec2 vectorFromCellPoint = pFrac - gridOffset - randomOffset;

			// Bell-shaped weight function
			float sqrDist = dot(vectorFromCellPoint, vectorFromCellPoint);
			float weight = exp(-sqrDist * 2.0);
			weight = max(0.0, weight - 0.01111);

			weightSum += weight;
			float waveInput = dot(vectorFromCellPoint, sideDir) + offset;

			// Add this cell's cosine and sine wave contributions
			phaseDir += vec2(cos(waveInput), sin(waveInput)) * weight;
		}
	}

	vec2  interpolated = phaseDir / weightSum;
	float magnitude = sqrt(dot(interpolated, interpolated));
	magnitude = max(1.0 - normalization, magnitude);
	return vec4(interpolated / magnitude, sideDir);
}

float gate(float min_val, float max_val, float val) {
	return val * smoothstep(min_val, max_val, val);
}

float threshold(float minv, float maxv, float width, float val) {
	// float halfWidth = width * 0.5;
	float halfWidth = 0.5 * min((maxv - minv), width);
	return val * smoothstep(minv - halfWidth, minv, val) * (1.0 - smoothstep(maxv, maxv + halfWidth, val));
}

float sustain(float minv, float maxv, float width, float val) {
	// float halfWidth = width * 0.5;
	float halfWidth = 0.5 * min(abs(maxv - minv), width);
	return smoothstep(minv - halfWidth, minv, val) * (1.0 - smoothstep(maxv, maxv + halfWidth, val));
}

float band(float minv, float maxv, float width, float val) {
	// float halfWidth = width * 0.5;
	float halfWidth = 0.5 * min(abs(maxv - minv), width);
	return smoothstep(minv - halfWidth, minv, val) * (1.0 - smoothstep(minv, minv + halfWidth, val)) +
		smoothstep(maxv - halfWidth, maxv, val) * (1.0 - smoothstep(maxv - halfWidth, maxv + halfWidth, val));
}

#define ADSR_FADE(t, start, attack, sustain, release)                                                                  \
	(smoothstep(start, start + attack, t) *                                                                            \
	 (1.0 - smoothstep(start + attack + sustain, start + attack + sustain + release, t)))

#define EVAL_LOD_OPTIMIZED(OUT_VAR, FUNC, TRANS_LEN, SEG_LEN, CUR_LEN)                                                 \
	{                                                                                                                  \
		float _layer = floor((CUR_LEN) / (SEG_LEN));                                                                   \
		float _local = mod((CUR_LEN), (SEG_LEN));                                                                      \
		float _blend = smoothstep((SEG_LEN) - (TRANS_LEN), (SEG_LEN), _local);                                         \
		OUT_VAR = FUNC(_layer);                                                                                        \
		if (_blend > 0.0) {                                                                                            \
			OUT_VAR = mix(OUT_VAR, FUNC(_layer + 1.0), _blend);                                                        \
		}                                                                                                              \
	}

// FUNC: A function that takes a float layer_index and returns your procedural texture (float, vec2, vec4, etc.)
#define LOD_BLEND(FUNC, TRANS_LEN, SEG_LEN, CUR_LEN)                                                                   \
	mix(FUNC(floor((CUR_LEN) / (SEG_LEN))),                                                                            \
	    FUNC(floor((CUR_LEN) / (SEG_LEN)) + 1.0),                                                                      \
	    smoothstep((SEG_LEN) - (TRANS_LEN), (SEG_LEN), mod((CUR_LEN), (SEG_LEN))))

float remap(float value, float valueMin, float valueMax) {
	return (value - valueMin) / (valueMax - valueMin);
}

float remapClamp(float value, float inMin, float inMax, float outMin, float outMax) {
	float t = clamp((value - inMin) / (inMax - inMin), 0.0, 1.0);
	return mix(outMin, outMax, t);
}

float adjust(float value, float scaly) {
	float f = 1.0 - value;
	float h = 0.4; // adjustable filter

	float a = scaly * (1.0 - h) + h;
	return clamp((remap(a, f, f + h)), 0.0, 1.0);
}

// https://iquilezles.org/articles/smin
float smin(float a, float b, float k) {
	float h = max(k - abs(a - b), 0.0);
	return min(a, b) - h * h * 0.25 / k;
}

float smaxCubic(float a, float b, float k) {
	k *= 1.4;
	float h = max(k - abs(a - b), 0.0);
	return max(a, b) + h * h * h / (6.0 * k * k);
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

float biome_map(float temperature, float moisture, float rocky) {
	return dot(vec3(temperature, moisture, rocky), vec3(0.2126, 0.7152, 0.0722));
}

// Assuming cross_noise_fbm and dot_noise_fbm from previous implementations are in scope
struct TerrainConfig {
    float spatial_scale;
    float min_height;
    float max_height;
    float ridge_weight;
	float biome_bleed;
};

float evaluate_terrain(vec3 p, float phase, float warp_strength, TerrainConfig config, out float out_biome, out float out_mask) {
	p *= config.spatial_scale;
    mat3 J_flow;
    vec3 unused_grad; // Placeholder for when you implement full analytical normals

    // 1. Evaluate the divergence-free vector field and its Jacobian
    vec3 flow = cross_noise_fbm(p, 2, phase, J_flow);

    // 2. Isolate the symmetric strain tensor (S)
    mat3 S = 0.5 * (J_flow + transpose(J_flow));

    // 3. Calculate tensor invariants
    // First invariant (I1) is the trace (divergence)
    float I1 = S[0][0] + S[1][1] + S[2][2];

    // For a symmetric matrix, tr(S^2) is the sum of squared elements
    float tr_S2 = dot(S[0], S[0]) + dot(S[1], S[1]) + dot(S[2], S[2]);

    // Second invariant (I2) yields a scalar representation of shear stress
    float I2 = 0.5 * (I1 * I1 - tr_S2);

    // float continent_mask = smoothstep(0.2, -0.5, I1);
    // float base_height = dot_noise_fbm(p, 4, phase, unused_grad) * continent_mask;
    // vec3 p_warped = p + (S * p) * warp_strength;

    // 4. Base Continent Mask
    // Map negative divergence (convergence) to 1.0 (land), positive to 0.0 (ocean/valleys)
    float continent_mask = 1.0 - smoothstep(-0.5, 0.2, I1);

    // Evaluate low-frequency baseline elevation
    float terrain_noise = dot_noise_fbm(p, 4, phase, unused_grad);
	float base_height = remap(terrain_noise, -1.0, 1.50*continent_mask, -1, 2.0*continent_mask);
    // 5. Anisotropic Domain Warping
    // Multiply p by the strain tensor to stretch the coordinate space along the principal axes of deformation
    vec3 p_warped = p+(S * p) * warp_strength;

    // 6. Shear-Guided High-Frequency Detail
    // Isolate areas of high shear stress using I2 to mask the jagged ridges
    float ridge_mask = smoothstep(0.0, 0.5, abs(I2)) * continent_mask;

    // Evaluate high-frequency noise using the warped domain, mapped to a sharp ridge function
    float raw_ridge = dot_noise_fbm(p_warped, 6, phase + 42.0, unused_grad);
    float ridge_height = (1.0 - remap(raw_ridge, -1, 1, 0, 1)) * ridge_mask; // Ridged multifractal style
    // float ridge_height = (1.0 - abs(raw_ridge)) * ridge_mask; // Ridged multifractal style

    float finalHeight = base_height + (ridge_height * 0.5);

	// Inside your evaluate_terrain function, after calculating S, I1, I2, and combined height:
	vec3 dominant_axis = normalize(S * vec3(1.0, 1.0, 1.0));

	// 1. Establish the base 3D biome coordinates
	float base_temp = 1.0 - finalHeight;
	float base_moisture = smoothstep(-1.0, 1.0, I1);
	float base_rocky = smoothstep(0.0, 0.8, abs(I2)); // Shear stress maps perfectly to rockiness

	vec3 biome_coords = vec3(base_temp, base_moisture, base_rocky);

	// 2. Anisotropic Bleed
	// Perturb all three parameters along the principal axis of tectonic deformation
	float biome_dither = raw_ridge * config.biome_bleed;
	biome_coords += dominant_axis * biome_dither;
	biome_coords = clamp(biome_coords, 0.0, 1.0);

	// 3. Resolve to the single float output
	out_biome = biome_map(biome_coords.x, biome_coords.y, biome_coords.z);
	out_mask = continent_mask;

	return remap(finalHeight, 0.0, 1.0, config.min_height, config.max_height);
}


vec3 evaluate_terrain_normal(vec3 p, float phase, float warp_strength, float eps, TerrainConfig config) {
    const vec2 k = vec2(1.0, -1.0);

    // Evaluate the terrain 4 times offset in a tetrahedron
	float b,m;
    float h1 = evaluate_terrain(p + k.xyy * eps, phase, warp_strength, config, b, m);
    float h2 = evaluate_terrain(p + k.yyx * eps, phase, warp_strength, config, b, m);
    float h3 = evaluate_terrain(p + k.yxy * eps, phase, warp_strength, config, b, m);
    float h4 = evaluate_terrain(p + k.xxx * eps, phase, warp_strength, config, b, m);

    // Accumulate the gradient vectors
    vec3 grad = k.xyy * h1 + k.yyx * h2 + k.yxy * h3 + k.xxx * h4;

    // Normalize to get the surface normal. grad.x/grad.z are proportional to +dHeight/dx,
    // +dHeight/dz (tetrahedron-gradient identity); the up-facing height-field normal needs
    // -dHeight/dx, -dHeight/dz (from Tx x Tz for a surface (x, H(x,z), z)), hence the negation.
    // The exact scaling of grad.y vs grad.xz depends on your world-space scale.
    return normalize(vec3(-grad.x, 2.0 * eps, -grad.z));
}
