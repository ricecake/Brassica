// A standard 32-bit integer hash
uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

uint fastVec3ToUint(vec3 p) {
    uvec3 x = floatBitsToUint(p);
    const uint q = 1103515245u; // Standard LCG multiplier

    x = ((x >> 8u) ^ x.yzx) * q;
    x = ((x >> 8u) ^ x.yzx) * q;
    x = ((x >> 8u) ^ x.yzx) * q;

    return x.x ^ x.y ^ x.z;
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

// High-quality 32-bit integer hash to generate deterministic pseudo-random seeds
uint hashUint(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}
