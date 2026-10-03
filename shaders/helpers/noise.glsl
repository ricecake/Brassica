#ifndef HELPERS_NOISE_GLSL
#define HELPERS_NOISE_GLSL

float mod289(const in float x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(const in vec2 x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 mod289(const in vec3 x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(const in vec4 x) {
	return x - floor(x * (1.0 / 289.0)) * 289.0;
}

float permute(const in float v) {
	return mod289(((v * 34.0) + 1.0) * v);
}

vec2 permute(const in vec2 v) {
	return mod289(((v * 34.0) + 1.0) * v);
}

vec3 permute(const in vec3 v) {
	return mod289(((v * 34.0) + 1.0) * v);
}

vec4 permute(const in vec4 v) {
	return mod289(((v * 34.0) + 1.0) * v);
}

float taylorInvSqrt(const in float r) {
	return 1.79284291400159 - 0.85373472095314 * r;
}

vec4 taylorInvSqrt(const in vec4 r) {
	return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise3d(vec3 v) {
	const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

	vec3 i = floor(v + dot(v, C.yyy));
	vec3 x0 = v - i + dot(i, C.xxx);

	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min(g.xyz, l.zxy);
	vec3 i2 = max(g.xyz, l.zxy);

	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy;
	vec3 x3 = x0 - D.yyy;

	i = mod289(i);
	vec4 p = permute(
		permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x +
		vec4(0.0, i1.x, i2.x, 1.0)
	);

	float n_ = 0.142857142857;
	vec3  ns = n_ * D.wyz - D.xzx;

	vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_);

	vec4 x = x_ * ns.x + ns.yyyy;
	vec4 y = y_ * ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);

	vec4 b0 = vec4(x.xy, y.xy);
	vec4 b1 = vec4(x.zw, y.zw);

	vec4 s0 = floor(b0) * 2.0 + 1.0;
	vec4 s1 = floor(b1) * 2.0 + 1.0;
	vec4 sh = -step(h, vec4(0.0));

	vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
	vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

	vec3 p0 = vec3(a0.xy, h.x);
	vec3 p1 = vec3(a0.zw, h.y);
	vec3 p2 = vec3(a1.xy, h.z);
	vec3 p3 = vec3(a1.zw, h.w);

	vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

	vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
	m = m * m;
	return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

vec3 hash33(vec3 p) {
	p = fract(p * vec3(443.897, 441.423, 437.195));
	p += dot(p, p.yxz + 19.19);
	return fract((p.xxy + p.yxx) * p.zyx);
}

vec2 hash22(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * vec3(443.897, 441.423, 437.195));
	p3 += dot(p3, p3.yzx + 19.19);
	return fract((p3.xx + p3.yz) * p3.zy);
}

float hash13Tile(in vec3 pos, in vec3 period) {
	if (period.x > 0.0) pos = mod(pos, period);
    pos  = fract(pos * vec3(.1031, .1030, .0973));
    pos += dot(pos, pos.zyx + 31.32);
    return fract((pos.x + pos.y) * pos.z);
}

float alligator(vec3 position, float gridsize, vec3 seed, bool tiling) {

    // Only int gridsize if it needs to tile
    if (tiling) gridsize = round(gridsize);
    gridsize = max(1.0, gridsize);

    position *= gridsize;
    position += hash33(vec3(seed)); // offset so FBM grids do not align

    vec3 id = floor(position);
    vec3 grid = position - id; // fract(position)

    float densest = 0.0;
    float secondDensest = 0.0;

    // Search the current cell and its neighbors (3x3x3)
    for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
    for (int z = -1; z <= 1; ++z) {

        vec3 offset = vec3(x, y, z);
        vec3 cell = id + offset;

        // Wrap cell coordinates to make the noise tile over [0,1]
        if (tiling) cell = mod(cell, gridsize);

         // Offset the hash input to avoid hashing zero coordinates
        cell += vec3(seed);

        // Get random center of the Cell
        vec3 center = hash33(cell) + offset;

        float dist = distance(grid, center);

        // 'if(dist < 1.0)' doesn't have any effect and doesn't improve
        // performance. See: https://www.shadertoy.com/view/MflGWM

        // Get random density scaled by the distance to the random point
        float density = hash13Tile(cell, vec3(gridsize)) * smoothstep(0, 1, 1.0 - dist);

        // Keep track of the two largest density values
        if (densest < density) {
            secondDensest = densest;
            densest = density;

        } else if (secondDensest < density) {
            secondDensest = density;
        }
    }
    // The alligator noise is the difference between the two largest densities
    return densest - secondDensest;
}

// Fractal Brownian motion applied to noise
float fractal(vec3 position, float gridsize, int octaves,
              float lacunarity, float persistence, bool tiling) {

    // For Amplitude math see: www.desmos.com/calculator/xgqeepapwn
    float amplitude = 1.0;
    float amplitudeSquaredSum = 0.0;
    float result = 0.0;
    vec3 seed = vec3(421); // can be any positive integer

    // Accumulate detail layers
    for(int i = 0; i < max(1, octaves); ++i) {
        // sample noise and apply amplitude
        result += alligator(position, gridsize, seed, tiling) * amplitude;
        amplitudeSquaredSum += amplitude * amplitude;

        // Increment for next iteration
        gridsize *= lacunarity;
        amplitude *= persistence;
        seed += gridsize;
    }

    // Restore variance lost by averaging
    // FabriceNeyret2 https://www.shadertoy.com/view/4dcSDr
    return result / sqrt(amplitudeSquaredSum);
}


#endif // HELPERS_NOISE_GLSL
