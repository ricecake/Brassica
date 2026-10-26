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

// float beerPowder(float d, float local_d) {
// 	// Approximation of multiple scattering (Beer-Powder law)
// 	// Ensuring sunny side isn't black when d is small
// 	return max(
// 		exp(-d),
// 		exp(-d * cloudPowderScale) * cloudPowderMultiplier * (1.0 - exp(-local_d * cloudPowderLocalScale))
// 	);
// }

float beerPowder(float d, float local_d) {
	// Approximation of multiple scattering (Beer-Powder law)
	// Ensuring sunny side isn't black when d is small
	return max(
		exp(-d),
		exp(-d * 0.1) * 50 * (1.0 - exp(-local_d * 2.67))
	);
}

// vec3 beerPowder(vec3 d, vec3 local_d) {
// 	// Approximation of multiple scattering (Beer-Powder law)
// 	// Ensuring sunny side isn't black when d is small
// 	return max(
// 		exp(-d),
// 		exp(-d * cloudPowderScale) * cloudPowderMultiplier * (vec3(1.0) - exp(-local_d * cloudPowderLocalScale))
// 	);
// }

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

float cross2d(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
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
    return 0.5 + 0.25 * sin(lobes * theta + offset);
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
};

OrganicPlateData getOrganicPlate(vec2 worldPos, float scale, float lobes) {
    vec2 p = worldPos * scale;

    // 1. Map Cartesian space to the specific Simplex Triangle
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

    // 2. Evaluate the function exactly on the 3 edges of this triangle
    vec2 p01 = getEdgePoint(v0_skew, v0, v1_skew, v1, lobes);
    vec2 p12 = getEdgePoint(v1_skew, v1, v2_skew, v2, lobes);
    vec2 p20 = getEdgePoint(v2_skew, v2, v0_skew, v0, lobes);

    // 3. Define the central convergence point
    vec2 center = (p01 + p12 + p20) / 3.0;

    // 4. Identify which of the 3 sub-regions the fragment falls into
    vec2 closest_v_skew = v0_skew;
    vec2 closest_v = v0;

    bool sign01_p = cross2d(p01 - center, p - center) > 0.0;
    bool sign01_v0 = cross2d(p01 - center, v0 - center) > 0.0;
    bool sign20_p = cross2d(p20 - center, p - center) > 0.0;
    bool sign20_v0 = cross2d(p20 - center, v0 - center) > 0.0;

    if (sign01_p == sign01_v0 && sign20_p == sign20_v0) {
        closest_v_skew = v0_skew;
        closest_v = v0;
    } else {
        bool sign12_p = cross2d(p12 - center, p - center) > 0.0;
        bool sign12_v1 = cross2d(p12 - center, v1 - center) > 0.0;
        bool sign01_v1 = cross2d(p01 - center, v1 - center) > 0.0;

        if (sign12_p == sign12_v1 && sign01_p == sign01_v1) {
            closest_v_skew = v1_skew;
            closest_v = v1;
        } else {
            closest_v_skew = v2_skew;
            closest_v = v2;
        }
    }

    // 5. Populate and return the struct
    OrganicPlateData data;
    data.regionVertex = closest_v_skew;
    data.regionID = hash21(closest_v_skew + 123.456);
    data.regionCenter = closest_v;
    data.nearestCentroid = center-(center.x+center.y)*G2;

    data.edgeDist = min(sdLine(p, center, p01),
                    min(sdLine(p, center, p12),
                        sdLine(p, center, p20)));

    return data;
}
