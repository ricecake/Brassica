
const float FAKE_PLANET_RADIUS = 600000.0; // 600km radius (1/10th scale planet)
const float PI = 3.14159265359;
const float PHI = 1.618033988749894848204586834;
const float TAU = 2.0 * PI;

const mat3 GOLD = mat3(
-0.571464913, +0.814921382, +0.096597072,
-0.278044873, -0.303026659, +0.911518454,
+0.772087367, +0.494042493, +0.399753815);

const int bayer4x4[16] = int[](
		0,  8,  2, 10,
	12,  4, 14,  6,
		3, 11,  1,  9,
	15,  7, 13,  5
);

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
	vec3 localDir = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
	vec3 up = abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
	vec3 tangent = normalize(cross(up, normal));
	vec3 bitangent = cross(normal, tangent);
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
		val += amp * dot_noise((p+(val/freq)) * freq, phase * freq);
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
    n = (n ^ (n <<  8u)) & 0x00ff00ffu; // n = ---- ---- fedc ba98 ---- ---- 7654 3210
    n = (n ^ (n <<  4u)) & 0x0f0f0f0fu; // n = ---- fedc ---- ba98 ---- 7654 ---- 3210
    n = (n ^ (n <<  2u)) & 0x33333333u; // n = --fe --dc --ba --98 --76 --54 --32 --10
    n = (n ^ (n <<  1u)) & 0x55555555u; // n = f e d c b a 9 8 7 6 5 4 3 2 1 0
    return n;
}

// Compacts every other bit of a 32-bit uint back into 16 contiguous bits
uint unpart1by1(uint n) {
    n &= 0x55555555u;                  // n = f e d c b a 9 8 7 6 5 4 3 2 1 0
    n = (n ^ (n >>  1u)) & 0x33333333u; // n = --fe --dc --ba --98 --76 --54 --32 --10
    n = (n ^ (n >>  2u)) & 0x0f0f0f0fu; // n = ---- fedc ---- ba98 ---- 7654 ---- 3210
    n = (n ^ (n >>  4u)) & 0x00ff00ffu; // n = ---- ---- fedc ba98 ---- ---- 7654 3210
    n = (n ^ (n >>  8u)) & 0x0000ffffu; // n = ---- ---- ---- ---- fedc ba98 7654 3210
    return n;
}

// ENCODE: Interleaves two 16-bit values into a 32-bit index
uint encodeMorton2D(uvec2 coords) {
    return part1by1(coords.x) | (part1by1(coords.y) << 1u);
}

// DECODE: Extracts two 16-bit coordinates from a 32-bit Morton code
uvec2 decodeMorton2D(uint code) {
    return uvec2(
        unpart1by1(code),
        unpart1by1(code >> 1u)
    );
}

uint mortonOwenScramble(uvec2 p, uint seed) {
	uint morton = encodeMorton2D(p);
	return owenScrambleBase4(morton, seed);
}

float mortonOwenThreshold(ivec2 uv, int FrameId) {
	// float temporalShift = fract(float(FrameId) * 0.61803398);
    uint code = mortonOwenScramble(uvec2(uv), uint(FrameId));
    return fract(uintBitsToFloat(code));// + temporalShift);
}

float mortonOwenThreshold(vec2 uv, int FrameId) {
    return mortonOwenThreshold(ivec2(uv*8192), FrameId);
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

float InterleavedGradientNoise(vec2 uv, int FrameId){
	// uv += float(FrameId)  * (vec2(47, 17) * 0.695f);
	//vec3 magic = vec3( 12.9898, 78.233, 43758.5453123 );
	const vec3 magic = vec3( 0.06711056f, 0.00583715f, 52.9829189f );
	float spatialJitter = fract(magic.z * fract(dot(uv, magic.xy)));
	float temporalShift = fract(float(FrameId) * 0.61803398);
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
            vec2 delta = offset - frac;
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
	return smoothstep(minv - halfWidth, minv, val)*(1.0-smoothstep(minv, minv+halfWidth, val)) + smoothstep(maxv - halfWidth, maxv, val)*(1.0 - smoothstep(maxv-halfWidth, maxv + halfWidth, val));
}

#define ADSR_FADE(t, start, attack, sustain, release) \
    (smoothstep(start, start + attack, t) * (1.0 - smoothstep(start + attack + sustain, start + attack + sustain + release, t)))


#define EVAL_LOD_OPTIMIZED(OUT_VAR, FUNC, TRANS_LEN, SEG_LEN, CUR_LEN) \
    { \
        float _layer = floor((CUR_LEN) / (SEG_LEN)); \
        float _local = mod((CUR_LEN), (SEG_LEN)); \
        float _blend = smoothstep((SEG_LEN) - (TRANS_LEN), (SEG_LEN), _local); \
        OUT_VAR = FUNC(_layer); \
        if (_blend > 0.0) { \
            OUT_VAR = mix(OUT_VAR, FUNC(_layer + 1.0), _blend); \
        } \
    }

// FUNC: A function that takes a float layer_index and returns your procedural texture (float, vec2, vec4, etc.)
#define LOD_BLEND(FUNC, TRANS_LEN, SEG_LEN, CUR_LEN) \
    mix( \
        FUNC(floor((CUR_LEN) / (SEG_LEN))), \
        FUNC(floor((CUR_LEN) / (SEG_LEN)) + 1.0), \
        smoothstep((SEG_LEN) - (TRANS_LEN), (SEG_LEN), mod((CUR_LEN), (SEG_LEN))) \
    )
