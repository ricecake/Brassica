float InterleavedGradientNoise(vec2 uv, int FrameId) {
	// uv += float(FrameId)  * (vec2(47, 17) * 0.695f);
	// vec3 magic = vec3( 12.9898, 78.233, 43758.5453123 );
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	float      spatialJitter = fract(magic.z * fract(dot(uv, magic.xy)));
	float      temporalShift = fract(float(FrameId) * 0.61803398);
	return fract(spatialJitter + temporalShift);
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

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
// Input range: [0, 1023]
uint expandBits3D(uint v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

// Compacts a 30-bit Morton code back into a 10-bit integer by extracting every 3rd bit.
uint compactBits3D(uint v) {
    v = v & 0x09249249u;
    v = (v | (v >>  2)) & 0x030C30C3u;
    v = (v | (v >>  4)) & 0x0300F00Fu;
    v = (v | (v >>  8)) & 0x030000FFu;
    v = (v | (v >> 16)) & 0x000003FFu;
    return v;
}

// ENCODER: Takes 3D spatial coordinates and returns a single 30-bit Morton code
uint encodeMorton3D(uvec3 coords) {
    uint xx = expandBits3D(coords.x);
    uint yy = expandBits3D(coords.y);
    uint zz = expandBits3D(coords.z);
    return xx | (yy << 1) | (zz << 2);
}

// DECODER: Takes a 30-bit Morton code and restores the original 3D coordinates
uvec3 decodeMorton3D(uint morton) {
    uint x = compactBits3D(morton);
    uint y = compactBits3D(morton >> 1);
    uint z = compactBits3D(morton >> 2);
    return uvec3(x, y, z);
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

const int bayer4x4[16] = int[](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
// Normalized [0, 1) ordered-dither threshold for screen pixel p -- for even (not clumpy) thinning
// against a continuous [0, 1] coverage value, e.g. discard if bayerDither4x4(gl_FragCoord.xy) >= coverage.
float bayerDither4x4(vec2 p) {
	ivec2 cell = ivec2(p) & ivec2(3);
	return float(bayer4x4[cell.y * 4 + cell.x]) / 16.0;
}

// Generates a 3D Bayer value in [0, 1) directly from coordinates
uint bayer3D(uvec3 pos) {
    // Interleave the bits of x, y, z to form the 3D Morton code
    uint morton = (expandBits3D(pos.z) << 2) |
                  (expandBits3D(pos.y) << 1) |
                   expandBits3D(pos.x);

    // Reverse all 32 bits
    uint reversed = bitfieldReverse(morton);
	return reversed;
    // // Normalize to [0, 1) by dividing by 2^32
    // return float(reversed) * 2.3283064365386963e-10;
}

float bayer4x4StepPhase(ivec2 pixel, int index) {
	return float(bayer4x4[((pixel.x & 3) * 4 + (pixel.y & 3)) % 16]) / 16.0;
}

