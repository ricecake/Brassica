#include "lygia/generative/random.glsl"
#include "noise/trig_noise.glsl"
#include "helpers/hash.glsl"
#include "helpers/distribution.glsl"

// Computes the quadratic smooth maximum and its analytical derivative weights.
// Returns a vec3:
//   res.x = smoothly blended max value
//   res.y = weight for derivative of 'a' (da/dx)
//   res.z = weight for derivative of 'b' (db/dx)
vec4 smax_quad_deriv(vec4 a, vec4 b, float k) {
    float h = clamp(0.5 + 0.5 * (a.x - b.x) / k, 0.0, 1.0);
    float val = mix(b.x, a.x, h) + k * h * (1.0 - h);
    return vec4(val, a.yzw * h + b.yzw * (1.0 - h));
}

float luminance(vec3 c) {
	return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float ease_out(float t) {
	float v = 1.0 - clamp01(t);
	return 1.0 - v * v;
}

float smooth_start(float t, float smoothing) {
	if (t >= smoothing)
		return t - 0.5 * smoothing;
	return 0.5 * t * t / smoothing;
}

float terraceSmooth(float h, float numSteps, float slopeCoarseness) {
	float stepId = floor(h * numSteps);
	float fractional = fract(h * numSteps);

	// Smooth the transition edge between steps
	// slopeCoarseness: 0.0 = perfectly sharp, 1.0 = completely smooth
	float edge = smoothstep(0.0, slopeCoarseness, fractional);

	return (stepId + edge) / numSteps;
}

float henyeyGreenstein(float g, float cosTheta) {
	float g2 = g * g;
	return (1.0 - g2) / (4.0 * PI * pow(max(0.0001, 1.0 + g2 - 2.0 * g * cosTheta), 1.5));
}

float remap(float value, float low1, float high1, float low2, float high2) {
	return low2 + (value - low1) * (high2 - low2) / max(0.0001, (high1 - low1));
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

// float cloudPhase(float cosTheta) {
// 	// Dual-lobe Henyey-Greenstein for forward and back scattering
// 	// Blended with a large isotropic component to ensure visibility at all angles
// 	float hg = mix(henyeyGreenstein(cloudPhaseG1, cosTheta), henyeyGreenstein(cloudPhaseG2, cosTheta), cloudPhaseAlpha);
// 	return mix(hg, (1.0 / (4.0 * PI)), cloudPhaseIsotropic);
// }

// Approximation of multiple scattering (Beer-Powder law). Ensures the sunny side isn't black
// when d is small. directScale/multiplier/localScale were hardcoded literals (0.1, 50, 2.67)
// until CloudLightingManager's directBeerPowderScale/cloudPowderMultiplier/
// cloudPowderLocalScale gave them a settings surface -- callers pass the UBO aliases
// (u_directBeerPowderScale etc., atmosphere/common.glsl) since this file has no UBO access
// of its own.
float beerPowder(float d, float local_d, float directScale, float multiplier, float localScale) {
	return max(
		exp(-d),
		exp(-d * directScale) * multiplier * (1.0 - exp(-local_d * localScale))
	);
}

// vec3 overload -- for a per-channel-extinction-tinted d/local_d (cloudExtinctionColor,
// CloudLightingManager), so the multi-scatter approximation stays consistently tinted with the
// rest of the optical-depth chain it's part of.
vec3 beerPowder(vec3 d, vec3 local_d, float directScale, float multiplier, float localScale) {
	return max(
		exp(-d),
		exp(-d * directScale) * multiplier * (vec3(1.0) - exp(-local_d * localScale))
	);
}

float schlickPhase(float cosTheta, float k) {
    float kCos = k * cosTheta;
    float denom = 1.0 - kCos;
    return 0.079577 * (1.0 - k * k) / (denom * denom);
}

float dualLobeSchlick(float cosTheta, float kFwd, float kBck, float weight) {
    return mix(schlickPhase(cosTheta, -kBck), schlickPhase(cosTheta, kFwd), weight);
}

struct HierarchicalWorleyData2D {
    float f1_dist;
    float f2_dist;
    vec2 f1_pos;
    vec2 f2_pos;
    float f1_level;
    float f2_level;
    float f1_id;
    float f2_id;
    vec2 p;
};

HierarchicalWorleyData2D hierarchicalWorleyTiled(in vec2 p, in float period) {
    vec2 n = floor(p);

    float f1_dist = 100.0;
    float f2_dist = 100.0;
    vec2 f1_pos = vec2(0.0);
    vec2 f2_pos = vec2(0.0);
    float f1_level = 0.0;
    float f2_level = 0.0;
    float f1_id = 0.0;
    float f2_id = 0.0;

    for(int j = -1; j <= 1; j++) {
        for(int i = -1; i <= 1; i++) {
            vec2 g1 = n + vec2(float(i), float(j));
            vec2 wrapped_g1 = mod(g1, period); // Tile Level 0

            vec3 rr = random3(wrapped_g1);
            vec2 o = g1 + rr.xy;
            vec2 r = p - o;
            float d = dot(r, r);
            float z = rr.z;

            if(z < 0.75) {
                float id = z + wrapped_g1.x + wrapped_g1.y * 7.0;
                if(d < f1_dist) {
                    f2_dist = f1_dist; f2_pos = f1_pos; f2_level = f1_level; f2_id = f1_id;
                    f1_dist = d; f1_pos = o; f1_level = 0.0; f1_id = id;
                } else if(d < f2_dist) {
                    f2_dist = d; f2_pos = o; f2_level = 0.0; f2_id = id;
                }
            } else {
                for(int l = 0; l < 2; l++) {
                    for(int k = 0; k < 2; k++) {
                        vec2 g2 = g1 + vec2(float(k), float(l)) / 2.0;
                        vec2 wrapped_g2 = mod(g2 * 2.0, period * 2.0); // Tile Level 1

                        rr = random3(wrapped_g2);
                        o = g2 + rr.xy / 2.0;
                        r = p - o;
                        d = dot(r, r);
                        z = rr.z;

                        if(z < 0.8) {
                            float id = z + wrapped_g2.x + wrapped_g2.y * 7.0;
                            if(d < f1_dist) {
                                f2_dist = f1_dist; f2_pos = f1_pos; f2_level = f1_level; f2_id = f1_id;
                                f1_dist = d; f1_pos = o; f1_level = 1.0; f1_id = id;
                            } else if(d < f2_dist) {
                                f2_dist = d; f2_pos = o; f2_level = 1.0; f2_id = id;
                            }
                        } else {
                            for(int ny = 0; ny < 2; ny++) {
                                for(int mx = 0; mx < 2; mx++) {
                                    vec2 g3 = g2 + vec2(float(mx), float(ny)) / 4.0;
                                    vec2 wrapped_g3 = mod(g3 * 4.0, period * 4.0); // Tile Level 2

                                    rr = random3(wrapped_g3);
                                    o = g3 + rr.xy / 4.0;
                                    r = p - o;
                                    d = dot(r, r);
                                    z = rr.z;

                                    if(z < 0.8) {
                                        float id = z + wrapped_g3.x + wrapped_g3.y * 7.0;
                                        if(d < f1_dist) {
                                            f2_dist = f1_dist; f2_pos = f1_pos; f2_level = f1_level; f2_id = f1_id;
                                            f1_dist = d; f1_pos = o; f1_level = 2.0; f1_id = id;
                                        } else if(d < f2_dist) {
                                            f2_dist = d; f2_pos = o; f2_level = 2.0; f2_id = id;
                                        }
                                    } else {
                                        for(int ty = 0; ty < 2; ty++) {
                                            for(int sx = 0; sx < 2; sx++) {
                                                vec2 g4 = g3 + vec2(float(sx), float(ty)) / 8.0;
                                                vec2 wrapped_g4 = mod(g4 * 8.0, period * 8.0); // Tile Level 3

                                                rr = random3(wrapped_g4);
                                                o = g4 + rr.xy / 8.0;
                                                r = p - o;
                                                d = dot(r, r);
                                                z = rr.z;

                                                float id = z + wrapped_g4.x + wrapped_g4.y * 7.0;
                                                if(d < f1_dist) {
                                                    f2_dist = f1_dist; f2_pos = f1_pos; f2_level = f1_level; f2_id = f1_id;
                                                    f1_dist = d; f1_pos = o; f1_level = 3.0; f1_id = id;
                                                } else if(d < f2_dist) {
                                                    f2_dist = d; f2_pos = o; f2_level = 3.0; f2_id = id;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return HierarchicalWorleyData2D(sqrt(f1_dist), sqrt(f2_dist), f1_pos, f2_pos, f1_level, f2_level, f1_id, f2_id, p);
}


struct OrganicMacroCell {
    vec2 corners[3];     // The base Cartesian vertices of the simplex triangle
    vec2 edgePoints[3];  // The displaced intersection points on the edges
    vec2 center;       // The central convergence point
};

// Internal helper: Evaluates the angular radius outward from a vertex
float _evaluateCellRadius(vec2 v_skew, vec2 v_cart, vec2 target_cart, float lobeCount) {
    vec2 dir = normalize(target_cart - v_cart);
    float theta = atan(dir.y, dir.x);

    // Hash function baked in to avoid external dependencies
    float hash = fract(sin(dot(v_skew, vec2(12.9898, 78.233))) * 43758.5453123);
    float offset = hash * 6.2831853;

    return 0.5 + 0.5 * sin(lobeCount * theta + offset);
}

// Internal helper: Finds the intersection ratio between two competing vertex radii
vec2 _calculateEdgeSplit(vec2 vA_skew, vec2 vA_cart, vec2 vB_skew, vec2 vB_cart, float lobeCount) {
    float rA = _evaluateCellRadius(vA_skew, vA_cart, vB_cart, lobeCount);
    float rB = _evaluateCellRadius(vB_skew, vB_cart, vA_cart, lobeCount);

    float t = rA / max(rA + rB, 0.0001);
    return mix(vA_cart, vB_cart, t);
}

// Public API: Generates the full 7-point geometry for a given simplex grid coordinate
OrganicMacroCell generateOrganicMacroCell(vec2 gridOriginSkew, float scale, float lobeCount) {
    OrganicMacroCell cell;

    const float G2 = 0.21132486540; // (3 - sqrt(3)) / 6

    // 1. Map skewed grid coordinates to scaled Cartesian space
    vec2 v0_skew = gridOriginSkew;
    vec2 v1_skew = gridOriginSkew + vec2(1.0, 0.0);
    vec2 v2_skew = gridOriginSkew + vec2(0.0, 1.0);

    cell.corners[0] = (v0_skew - (v0_skew.x + v0_skew.y) * G2) * scale;
    cell.corners[1] = (v1_skew - (v1_skew.x + v1_skew.y) * G2) * scale;
    cell.corners[2] = (v2_skew - (v2_skew.x + v2_skew.y) * G2) * scale;

    // 2. Evaluate the specific organic intersections along the 3 edges
    cell.edgePoints[0] = _calculateEdgeSplit(v0_skew, cell.corners[0], v1_skew, cell.corners[1], lobeCount);
    cell.edgePoints[1] = _calculateEdgeSplit(v1_skew, cell.corners[1], v2_skew, cell.corners[2], lobeCount);
    cell.edgePoints[2] = _calculateEdgeSplit(v2_skew, cell.corners[2], v0_skew, cell.corners[0], lobeCount);

    // 3. Define the center to guarantee a planar interior partition
    cell.center = (cell.edgePoints[0] + cell.edgePoints[1] + cell.edgePoints[2]) / 3.0;

    return cell;
}

// --- Required Helper Functions ---

// Computes barycentric coordinates (u, v, w) from vertex positions and distances to them
vec3 getBarycentricFromDistances(vec3 p, vec3 A, vec3 B, vec3 C) {
    float dA = distance(p, A);
    float dB = distance(p, B);
    float dC = distance(p, C);
    // 1. Compute side lengths squared
    float a2 = dot(B - C, B - C); // BC squared
    float b2 = dot(A - C, A - C); // AC squared
    float c2 = dot(A - B, A - B); // AB squared

    // 2. Square the given vertex distances
    float dA2 = dA * dA;
    float dB2 = dB * dB;
    float dC2 = dC * dC;

    // 3. Set up the 3x3 linear system: M * uvw = rhs
    // Row 0: u + v + w = 1
    // Row 1 & 2 come from distance differences:
    // (d_A^2 - d_C^2) and (d_B^2 - d_C^2)
    mat3 M = mat3(
        vec3(1.0, -b2,      -c2),                // Column 0 (u coefficients)
        vec3(1.0, c2 - a2,  c2),                 // Column 1 (v coefficients)
        vec3(1.0, b2,       a2 - b2)             // Column 2 (w coefficients)
    );

    vec3 rhs = vec3(
        1.0,
        dA2 - dC2 - b2,
        dB2 - dC2 - c2
    );

    // 4. Solve the system by multiplying by the inverse matrix
    // GLSL's inverse() is highly optimized and branchless hardware-side
    return inverse(M) * rhs;
}

// Computes barycentric coordinates (u, v, w) from 2D vertices and distances to them
vec3 getBarycentricFromDistances2D(vec2 p, vec2 A, vec2 B, vec2 C) {
    float dA = distance(p, A);
    float dB = distance(p, B);
    float dC = distance(p, C);
    // 1. Compute side lengths squared using native 2D dot products
    vec2 AB = B - A;
    vec2 AC = C - A;
    vec2 BC = C - B;

    float c2 = dot(AB, AB); // Side c^2 (AB)
    float b2 = dot(AC, AC); // Side b^2 (AC)
    float a2 = dot(BC, BC); // Side a^2 (BC)

    // 2. Square the incoming distances
    float dA2 = dA * dA;
    float dB2 = dB * dB;
    float dC2 = dC * dC;

    // 3. Set up a 2x2 system for u and v by subtracting the dC^2 equation
    // Row 1: (2*b^2)u       + (b^2 + c^2 - a^2)v = dA2 - dC2 + b^2
    // Row 2: (a^2 + b^2 - c^2)u + (2*a^2)v       = dB2 - dC2 + a^2
    mat2 M = mat2(
        vec2(2.0 * b2,         a2 + b2 - c2), // Column 0 (u coefficients)
        vec2(b2 + c2 - a2,     2.0 * a2)      // Column 1 (v coefficients)
    );

    vec2 rhs = vec2(
        dA2 - dC2 + b2,
        dB2 - dC2 + a2
    );

    // 4. Solve the 2x2 system analytically using matrix inversion
    // GLSL handles 2x2 inverse operations extremely efficiently in hardware
    vec2 uv = inverse(M) * rhs;

    // 5. Recover w using the normalization constraint
    float w = 1.0 - uv.x - uv.y;

    return vec3(uv.x, uv.y, w);
}

// Standard 2D Cramer's Rule Area Barycentrics (Recommended)
vec3 getBarycentric2D(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = b - a;
    vec2 v1 = c - a;
    vec2 v2 = p - a;

    float det = v0.x * v1.y - v1.x * v0.y;

    // Protect against degenerate zero-area triangles
    if (abs(det) < 0.00001) return vec3(0.3333);

    float v = (v2.x * v1.y - v1.x * v2.y) / det;
    float w = (v0.x * v2.y - v2.x * v0.y) / det;

    return vec3(1.0 - v - w, v, w);
}

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
}

float sdLine(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float getRadius(vec2 v_skew, vec2 v_cart, vec2 target_cart, float lobes) {
    vec2 dir = target_cart - v_cart;
    float theta = atan(dir.y, dir.x);
    float offset = hash21(v_skew) * 6.2831853;
    return 0.5 + 0.25 * sin(lobes *theta + offset);
}

vec2 getEdgePoint(vec2 vA_skew, vec2 vA, vec2 vB_skew, vec2 vB, float lobes) {
    float rA = getRadius(vA_skew, vA, vB, lobes);
    float rB = getRadius(vB_skew, vB, vA, lobes);
    float t = rA / max(rA + rB, 0.0001);
    return mix(vA, vB, t);
}

struct OrganicPlateData {
    vec2 regionVertex; // The skewed grid coordinate identifying the plate
    float regionID;    // A 0.0 - 1.0 hash unique to this specific plate
    float edgeDist;    // The explicit distance to the generated straight-line boundary
    vec2 regionCenter;
    vec2 nearestCentroid;
    vec3 barycentric;
};

// Assumes cross2d(a, b) is defined as (a.x * b.y - a.y * b.x)

// Assumes cross2d(a, b) is defined as (a.x * b.y - a.y * b.x)

// Safely hashes integer coordinates to exactly 0, 1, or 2,
// resolving negative wrapping issues safely.
int hash3(vec2 v) {
    int sum = int(v.x) + int(v.y);
    return (sum % 3 + 3) % 3;
}

// Assumes cross2d(a, b) is defined as (a.x * b.y - a.y * b.x)
OrganicPlateData getOrganicPlate(vec2 worldPos, float scale, float lobes) {
    vec2 p = worldPos * scale;

    const float F2 = 0.36602540378;
    const float G2 = 0.21132486540;

    vec2 s = p + (p.x + p.y) * F2;
    vec2 i = floor(s);
    vec2 f = s - i;

    vec2 i1 = (f.x > f.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    vec2 v0_skew = i;
    vec2 v1_skew = i + i1;
    vec2 v2_skew = i + vec2(1.0, 1.0);

    vec2 v0 = v0_skew - (v0_skew.x + v0_skew.y) * G2;
    vec2 v1 = v1_skew - (v1_skew.x + v1_skew.y) * G2;
    vec2 v2 = v2_skew - (v2_skew.x + v2_skew.y) * G2;

    vec2 V_skew[3];
    vec2 V_cart[3];

    V_skew[hash3(v0_skew)] = v0_skew; V_cart[hash3(v0_skew)] = v0;
    V_skew[hash3(v1_skew)] = v1_skew; V_cart[hash3(v1_skew)] = v1;
    V_skew[hash3(v2_skew)] = v2_skew; V_cart[hash3(v2_skew)] = v2;

    vec2 E01 = getEdgePoint(V_skew[0], V_cart[0], V_skew[1], V_cart[1], lobes);
    vec2 E12 = getEdgePoint(V_skew[1], V_cart[1], V_skew[2], V_cart[2], lobes);
    vec2 E20 = getEdgePoint(V_skew[2], V_cart[2], V_skew[0], V_cart[0], lobes);

    vec2 center = (E01 + E12 + E20) / 3.0;

    // Test exactly which of the 4 sub-triangles 'p' is inside
    bool inC0 = cross2d(E01 - E20, p - E20) * cross2d(E01 - E20, V_cart[0] - E20) >= 0.0;
    bool inC1 = cross2d(E12 - E01, p - E01) * cross2d(E12 - E01, V_cart[1] - E01) >= 0.0;
    bool inC2 = cross2d(E20 - E12, p - E12) * cross2d(E20 - E12, V_cart[2] - E12) >= 0.0;

    int tri_id = -1;
    if (inC0) tri_id = 0;
    else if (inC1) tri_id = 1;
    else if (inC2) tri_id = 2;

    vec2 pA = E01;
    vec2 pB = E12;
    vec2 pC = E20;

    vec2 active_v_skew;
    vec2 active_v;

    if (tri_id == 0) {
        pB = V_cart[0];
        active_v_skew = V_skew[0];
        active_v = V_cart[0];
    } else if (tri_id == 1) {
        pC = V_cart[1];
        active_v_skew = V_skew[1];
        active_v = V_cart[1];
    } else if (tri_id == 2) {
        pA = V_cart[2];
        active_v_skew = V_skew[2];
        active_v = V_cart[2];
    } else {
        float d0 = dot(p - V_cart[0], p - V_cart[0]);
        float d1 = dot(p - V_cart[1], p - V_cart[1]);
        float d2 = dot(p - V_cart[2], p - V_cart[2]);
        int closest_id = (d0 < d1 && d0 < d2) ? 0 : ((d1 < d2) ? 1 : 2);

        active_v_skew = V_skew[closest_id];
        active_v = V_cart[closest_id];
    }

    OrganicPlateData data;
    data.regionVertex = active_v_skew;
    data.regionID = hash21(active_v_skew + 123.456);
    data.regionCenter = active_v;
    data.nearestCentroid = center;

    // Pass the globally stable slots to the corrected barycentric solver
    data.barycentric = getBarycentric2D(p, pA, pB, pC);

    data.edgeDist = min(sdLine(p, center, E01),
                    min(sdLine(p, center, E12),
                        sdLine(p, center, E20)));

    return data;
}

vec3 hash_and_rotate(ivec3 grid_p, vec2 osc, int tile_size) {
    // Perfect tiling using bitmasking
    grid_p &= (tile_size - 1);

    uvec3 u = uvec3(grid_p);
    u.x = u.x * 1664525u + 1013904223u;
    u.y = u.y * 1664525u + 1013904223u;
    u.z = u.z * 1664525u + 1013904223u;
    u.x += u.y * u.z; u.y += u.x * u.z; u.z += u.x * u.y;
    u ^= u >> 16u;

    vec3 grad = vec3(u) * (1.0 / float(0xffffffffu)) * 2.0 - 1.0;
    grad = normalize(grad);

    // Internal Flow Noise boiling (replaces dual-evaluation Shepard tones)
    return vec3(
        grad.x * osc.x - grad.y * osc.y,
        grad.x * osc.y + grad.y * osc.x,
        grad.z
    );
}


void eval_simplex_node(ivec3 grid_p, vec3 x, vec2 osc, int tile_size, inout float val, inout vec3 grad) {
    float d2 = dot(x, x);
    if (d2 < 0.6) {
        vec3 g = hash_and_rotate(grid_p, osc, tile_size);
        float w = 0.6 - d2;
        float w3 = w * w * w;
        float w4 = w3 * w;
        float dot_gx = dot(g, x);

        val += w4 * dot_gx;

        grad += w4 * g - 8.0 * w3 * dot_gx * x;
    }
}

vec4 simplex_flow_noise_grad(vec3 p, vec2 osc, int tile_size) {
    const float F3 = 0.333333333;
    const float G3 = 0.166666667;

    vec3 s = floor(p + dot(p, vec3(F3)));
    vec3 x0 = p - s + dot(s, vec3(G3));

    vec3 g = step(x0.yzx, x0);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + G3;
    vec3 x2 = x0 - i2 + 2.0 * G3;
    vec3 x3 = x0 - 1.0 + 3.0 * G3;

    ivec3 grid_base = ivec3(s);
    float val = 0.0; vec3 grad = vec3(0.0);

    eval_simplex_node(grid_base, x0, osc, tile_size, val, grad);
    eval_simplex_node(grid_base + ivec3(i1), x1, osc, tile_size, val, grad);
    eval_simplex_node(grid_base + ivec3(i2), x2, osc, tile_size, val, grad);
    eval_simplex_node(grid_base + ivec3(1), x3, osc, tile_size, val, grad);

    return vec4(val, grad) * 31.316;
}

float simplex_fbm_curl(vec3 p, int oct, float phase, vec3 dir, float footprint, int tile_size) {
    float val = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;

    vec3 warp_curl = vec3(0.0);
    float max_nyquist_freq = 0.5 / max(footprint, 1e-7);
    vec3 curl_axis = normalize(dir);

    // Initia
    vec2 osc = vec2(cos(phase), sin(phase));
    float epsilon = 0.5;

    for (int i = 0; i < 16; i++) {
        if (i >= oct) break;

        float nyquist_fade = max(0.0, 1.0 - (freq / max_nyquist_freq));
        if (nyquist_fade <= 0.0) break;

        float current_amp = amp * nyquist_fade;
        float curl_len = length(warp_curl);

        // ALU-only Algebraic Pushbacks
        float theoretical_freq = freq * (1.0 + curl_len * 0.5);
        float current_freq = (max_nyquist_freq * theoretical_freq) / (max_nyquist_freq + theoretical_freq);

        float max_warp = 0.5 / current_freq;
        float safe_max_warp = max(max_warp, 1e-7);
        float bounded_warp = (safe_max_warp * curl_len) / (safe_max_warp + curl_len);

        vec3 warp_vec = curl_len > 0.0 ? (warp_curl / curl_len) * bounded_warp : vec3(0.0);
        vec3 warped_p = p + warp_vec;

        // Evolve oscillator for the next octave
        osc.x -= epsilon * osc.y;
        osc.y += epsilon * osc.x;

        vec4 n = simplex_flow_noise_grad(warped_p * current_freq, osc, tile_size);
        vec3 current_curl = cross(n.yzw, curl_axis);

        val += current_amp * n.x;
        warp_curl += current_curl * current_amp;

        max_amp += current_amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return val / max_amp;
}


// Organic Simplex Phasor Noise with Phase Advection
// Drop directly into Shadertoy's Image tab.

// --- Hashes & Base Noise ---
vec3 hash32(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Cheap 2D value noise for organic jitter and masking
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i + vec2(0.0,0.0)), hash12(i + vec2(1.0,0.0)), u.x),
               mix(hash12(i + vec2(0.0,1.0)), hash12(i + vec2(1.0,1.0)), u.x), u.y);
}

// Assumes a period of 1.0 instead of 2 * PI
vec4 poly_wave(vec4 phase) {
    vec4 x = fract(phase) * 2.0 - 1.0;
    vec4 y = 1.0 - (x * x);
    return (y * y) * 2.0 - 1.0;
}

// --- Differentiable Polynomial Oscillator ---
float polyWave(float phase) {
    phase = fract(phase);
    float t = abs(fract(phase + 0.5) * 2.0 - 1.0);
    return t * t * (3.0 - 2.0 * t) * 2.0 - 1.0;
}

// --- Vector Fields ---
// 1. Spatial frequency (Direction and tightness of the ridges)
vec2 getVectorField(vec2 p) {
    return vec2(-sin(p.y * 0.3), cos(p.x * 0.3)) * 3.0;
}

// 2. Wind Velocity (Direction and speed the waves drift)
vec2 getWindVector(vec2 p) {
    // A slow, non-uniform drift moving generally up and right
    return vec2(0.5 + sin(p.y * 0.1)*0.2, 0.5 + cos(p.x * 0.2)*0.2);
}

// --- Kernel Evaluation ---
vec2 evaluatePhasorKernel(vec2 gridCell, vec2 d, vec2 worldP, vec2 fieldDir, float phase) {
    vec3 h = hash32(gridCell);

    vec2 jitter = (h.xy - 0.5) * 0.4;
    vec2 kernelPos = d - jitter;

    float distSq = dot(kernelPos, kernelPos);
    float A = max(0.0, 0.5 - distSq);
    if (A <= 0.0) return vec2(0.0);
    A = A * A * A * A; // C2 continuous amplitude falloff

    vec2 kernelWorld = worldP - kernelPos;

    // ORGANIC VECTOR 2: Amplitude Masking (Patchiness)
    // Evaluated at kernel center so it doesn't break the amplitude envelope bounds
    float mask = vnoise(kernelWorld * 0.2);
    mask = smoothstep(0.3, 0.7, mask); // Sharpen to create distinct patches
    A *= mask;

    // Evaluate spatial frequency and wind velocity
    vec2 F = fieldDir;//getVectorField(kernelWorld);
    vec2 W = getWindVector(kernelWorld);

    // ORGANIC VECTOR 1: Phase Jitter (Micro-Wobble)
    // Low frequency structural wobble injected directly into the phase
    float phaseJitter = (vnoise(kernelWorld * 1.5) - 0.5) * 1.5;

    float phi0 = h.z;

    // PHASE ADVECTION: dot(F, W) is the temporal frequency.
    // Subtracting it from phase slides the wave along the wind vector continuously.
    float phi = dot(F, kernelPos) + phi0 + phaseJitter - (dot(F, W) * phase * 0.5);

    return A * vec2(polyWave(phi), polyWave(phi - 0.25));
}

float phasimpleNoise(vec2 uv, vec2 fieldDir, float phase) {
    // Simplex Skewing
    const float F2 = 0.366025403;
    const float G2 = 0.211324865;

    vec2 skew = uv + (uv.x + uv.y) * F2;
    vec2 cell = floor(skew);
    vec2 internal_pos = skew - cell;

    // Simplex Routing
    vec2 v1 = step(internal_pos.yx, internal_pos.xy);

    vec2 p0 = uv - (cell - (cell.x + cell.y) * G2);
    vec2 p1 = p0 - v1 + G2;
    vec2 p2 = p0 - 1.0 + 2.0 * G2;

    // Complex Summation
    vec2 Z = vec2(0.0);
    Z += evaluatePhasorKernel(cell, p0, uv, fieldDir, phase);
    Z += evaluatePhasorKernel(cell + v1, p1, uv, fieldDir, phase);
    Z += evaluatePhasorKernel(cell + 1.0, p2, uv, fieldDir, phase);

    // ORGANIC VECTOR 3: Algebraic Profile Shaping
    // 1. Normalize the continuous field to cleanly isolate the wave from kernel overlap intensities
    float normalized_wave = Z.x / max(length(Z), 0.0001);

    // 2. Map from [-1.0, 1.0] to [0.0, 1.0]
    float unshaped_ridges = normalized_wave * 0.5 + 0.5;

    // 3. Power curve for sharp asymmetrical peaks (x^4)
    float organic_ridges = unshaped_ridges * unshaped_ridges;
    organic_ridges *= organic_ridges;
    return organic_ridges;
}


// 3D to 4D hash: xyz for spatial jitter, w for initial phase
vec4 hash43(vec3 p) {
    vec4 p4 = fract(vec4(p.xyzx)  * vec4(0.1031, 0.1030, 0.0973, 0.1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

// float polyWave(float phase) {
//     phase = fract(phase);
//     float t = abs(fract(phase + 0.5) * 2.0 - 1.0);
//     return t * t * (3.0 - 2.0 * t) * 2.0 - 1.0;
// }

// Differentiable polynomial oscillator returning value and derivative d(polyWave)/d(phase)
float polyWaveDeriv(float phase, out float dWave) {
    phase = fract(phase);
    float t = abs(fract(phase + 0.5) * 2.0 - 1.0);
    float s = (fract(phase + 0.5) < 0.5) ? -2.0 : 2.0; // derivative of |...| wrt phase
    // t = |2 * fract(phase + 0.5) - 1|
    // poly = (t^2 * (3 - 2t)) * 2 - 1 = 6*t^2 - 4*t^3 - 1
    // d(poly)/dt = 12*t - 12*t^2 = 12 * t * (1 - t)
    // d(poly)/d(phase) = 12 * t * (1 - t) * s
    dWave = 12.0 * t * (1.0 - t) * s;
    return (t * t * (3.0 - 2.0 * t)) * 2.0 - 1.0;
}

// Internal kernel evaluator
vec2 evalPhasorKernel3D(vec3 cell, vec3 d, float mask, vec3 F, vec3 W, float phaseJitter, float time) {
    vec4 h = hash43(cell);

    // Jitter kernel center. Bound to 0.4 to stay within the 0.6 simplex radius
    vec3 jitter = (h.xyz - 0.5) * 0.4;
    vec3 kernelPos = d - jitter;

    // 3D Simplex boundary is 0.6
    float distSq = dot(kernelPos, kernelPos);
    float tA = max(0.0, 0.6 - distSq);
    if (tA <= 0.0) return vec2(0.0);

    // Apply C2 falloff and inject the external mask
    float A = tA * tA * tA * tA * mask;

    float phi0 = h.w;
    float phi = dot(F, kernelPos) + phi0 + phaseJitter - (dot(F, W) * time);

    return A * vec2(polyWave(phi), polyWave(phi - 0.25));
}

// Internal kernel evaluator with analytical gradient
vec2 evalPhasorKernel3DGrad(vec3 cell, vec3 d, float mask, vec3 F, vec3 W, float phaseJitter, float time, out vec3 kernelGrad) {
    vec4 h = hash43(cell);

    // Jitter kernel center. Bound to 0.4 to stay within the 0.6 simplex radius
    vec3 jitter = (h.xyz - 0.5) * 0.4;
    vec3 kernelPos = d - jitter;

    // 3D Simplex boundary is 0.6
    float distSq = dot(kernelPos, kernelPos);
    float tA = max(0.0, 0.6 - distSq);
    if (tA <= 0.0) {
        kernelGrad = vec3(0.0);
        return vec2(0.0);
    }

    // A = tA^4 * mask where tA = (0.6 - distSq)
    // dA/d(kernelPos) = 4 * tA^3 * (-2 * kernelPos) * mask = -8 * kernelPos * tA^3 * mask
    float tA3 = tA * tA * tA;
    float A = tA * tA3 * mask;
    vec3 gradA = -8.0 * kernelPos * tA3 * mask;

    float phi0 = h.w;
    float phi = dot(F, kernelPos) + phi0 + phaseJitter - (dot(F, W) * time);

    // For real part (polyWave(phi)):
    float dWave;
    float wave = polyWaveDeriv(phi, dWave);

    // grad(A * wave) = gradA * wave + A * dWave * grad(phi)
    // Since phi = dot(F, kernelPos) + ..., grad(phi) = F
    kernelGrad = gradA * wave + (A * dWave) * F;

    return A * vec2(wave, polyWave(phi - 0.25));
}

// Main 3D entry point
vec2 phasorNoise3D(vec3 p, float mask, vec3 F, vec3 W, float phaseJitter, float time) {
    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;

    // 1. Simplex Skewing
    vec3 skew = p + dot(p, vec3(F3));
    vec3 cell = floor(skew);
    vec3 internal_pos = skew - cell;

    // 2. Simplex Routing
    vec3 g = step(internal_pos.yzx, internal_pos.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    // Unskew
    vec3 p0 = p - cell + dot(cell, vec3(G3));
    vec3 p1 = p0 - i1 + vec3(G3);
    vec3 p2 = p0 - i2 + vec3(G3 * 2.0);
    vec3 p3 = p0 - 1.0 + vec3(G3 * 3.0);

    // 3. Summation (4 kernels instead of 27 for Cartesian)
    vec2 Z = vec2(0.0);
    Z += evalPhasorKernel3D(cell, p0, mask, F, W, phaseJitter, time);
    Z += evalPhasorKernel3D(cell + i1, p1, mask, F, W, phaseJitter, time);
    Z += evalPhasorKernel3D(cell + i2, p2, mask, F, W, phaseJitter, time);
    Z += evalPhasorKernel3D(cell + 1.0, p3, mask, F, W, phaseJitter, time);

    return Z;
}

// 3D Simplex Phasor Noise with Analytical Gradient
vec2 phasorNoise3DGrad(vec3 p, float mask, vec3 F, vec3 W, float phaseJitter, float time, out vec3 grad) {
    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;

    // 1. Simplex Skewing
    vec3 skew = p + dot(p, vec3(F3));
    vec3 cell = floor(skew);
    vec3 internal_pos = skew - cell;

    // 2. Simplex Routing
    vec3 g = step(internal_pos.yzx, internal_pos.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    // Unskew
    vec3 p0 = p - cell + dot(cell, vec3(G3));
    vec3 p1 = p0 - i1 + vec3(G3);
    vec3 p2 = p0 - i2 + vec3(G3 * 2.0);
    vec3 p3 = p0 - 1.0 + vec3(G3 * 3.0);

    // 3. Summation and Gradient Accumulation
    vec2 Z = vec2(0.0);
    grad = vec3(0.0);

    vec3 g0, g1, g2, g3;
    Z += evalPhasorKernel3DGrad(cell, p0, mask, F, W, phaseJitter, time, g0);
    Z += evalPhasorKernel3DGrad(cell + i1, p1, mask, F, W, phaseJitter, time, g1);
    Z += evalPhasorKernel3DGrad(cell + i2, p2, mask, F, W, phaseJitter, time, g2);
    Z += evalPhasorKernel3DGrad(cell + 1.0, p3, mask, F, W, phaseJitter, time, g3);

    grad = g0 + g1 + g2 + g3;

    // The normalized value is V = Z.x / |Z|.
    // dV/dp = d(Z.x / length(Z)) = grad / |Z| - Z.x * (Z.x * grad) / |Z|^3
    float lenZ = length(Z);
    if (lenZ > 0.0001) {
        grad = (grad - (Z.x / (lenZ * lenZ)) * grad) / lenZ;
    } else {
        grad = vec3(0.0);
    }

    return Z;
}

struct NoiseCascade {
    vec3 frequencies;
    vec3 weights;
};

// Calculates the weight for a single layer based on its relative log-distance
float getLayerWeight(float relativeDist, float fadeIn, float hold, float fadeOut) {
    float absDist = abs(relativeDist);
    if (absDist <= hold * 0.5) return 1.0;

    float fadeStart = hold * 0.5;
    float fadeEnd = fadeStart + (relativeDist < 0.0 ? fadeIn : fadeOut);

    return 1.0 - smoothstep(fadeStart, fadeEnd, absDist);
}

NoiseCascade calculateDynamicLOD(float dist, float baseFreq, float growthFactor) {
    // Map distance to continuous logarithmic octave scale
    float continuousLOD = log2(max(dist, 1.0)) / log2(growthFactor);

    float baseLayer = floor(continuousLOD);
    float t = fract(continuousLOD); // 0.0 to 1.0 transition within current octave

    NoiseCascade cascade;

    // Frequencies for Macro (baseLayer - 1), Medium (baseLayer), Micro (baseLayer + 1)
    cascade.frequencies = vec3(
        baseFreq * pow(growthFactor, -(baseLayer - 1.0)),
        baseFreq * pow(growthFactor, -baseLayer),
        baseFreq * pow(growthFactor, -(baseLayer + 1.0))
    );

    // Absolute octave-space distance from continuousLOD to each layer's integer index
    vec3 octaveDistances = vec3(
        1.0 + t, // Layer -1 (Macro)
        t,       // Layer 0  (Medium)
        1.0 - t  // Layer +1 (Micro)
    );

    // Raised Cosine (Hann) window ensures smooth 3-tap blending with zero gaps.
    // Weights are 1.0 at center distance 0.0, and smoothly curve down to 0.0 at distance 1.0.
    const float PI = 3.14159265;
    cascade.weights = vec3(
        octaveDistances.x < 1.0 ? 0.5 + 0.5 * cos(octaveDistances.x * PI) : 0.0,
        octaveDistances.y < 1.0 ? 0.5 + 0.5 * cos(octaveDistances.y * PI) : 0.0,
        octaveDistances.z < 1.0 ? 0.5 + 0.5 * cos(octaveDistances.z * PI) : 0.0
    );

    // Variance normalization preserves contrast and amplitude across blends
    float weightSumSq = dot(cascade.weights, cascade.weights);
    if (weightSumSq > 0.0) {
        cascade.weights *= inversesqrt(weightSumSq);
    }

    return cascade;
}
