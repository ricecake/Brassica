#include "common.glsl"
#include "terrain.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "lygia/generative/noised.glsl"
#include "helpers/wind.glsl"

struct TerrainConfig {
    float spatial_scale;
    float min_height;
    float max_height;
    float ridge_weight;
    float biome_bleed;
};

struct TectonicPlate {
    vec3 seed_dir;
    float height;
    float k;
    vec3 velocity;
};

const TectonicPlate plates[] = TectonicPlate[](
    TectonicPlate(vec3(-0.10998967, 0.18729015, 0.9761274), -0.2120724, 45.14807, vec3(0.77386445, -0.6001565, 0.20235115)),
    TectonicPlate(vec3(-0.27727792, 0.5709581, 0.77273786), 0.2528148, 58.432438, vec3(0.3304707, -0.69852555, 0.6347056)),
    TectonicPlate(vec3(0.51554155, 0.5044243, -0.6926565), -0.2947329, 51.651794, vec3(-0.7036822, 0.71048665, -0.006338941)),
    TectonicPlate(vec3(0.40002212, -0.9128037, -0.082290396), -0.21447644, 30.138618, vec3(-0.3685467, -0.078000076, -0.9263311)),
    TectonicPlate(vec3(0.83764374, -0.41415262, -0.35613286), -0.20410867, 31.204689, vec3(0.43418014, 0.10924039, 0.8941779)),
    TectonicPlate(vec3(-0.24284463, 0.95516706, 0.16935897), -0.20179172, 32.290123, vec3(-0.9077131, -0.16216363, -0.3869881)),
    TectonicPlate(vec3(-0.25884876, -0.95428, -0.14948884), 0.2607334, 30.138618, vec3(-0.36568236, 0.2400595, -0.89924854)),
    TectonicPlate(vec3(0.584477, -0.523526, 0.61992514), 0.20748353, 22.928429, vec3(0.28582802, -0.5821977, -0.76114917)),
    TectonicPlate(vec3(0.56173223, 0.8076131, 0.1794939), -0.20347077, 23.187391, vec3(-0.82534325, 0.5320569, 0.18900792)),
    TectonicPlate(vec3(-0.8119338, 0.57554865, 0.09750533), 0.20122617, 29.046818, vec3(-0.10501896, 0.020289334, -0.99426323)),
    TectonicPlate(vec3(-0.7352156, -0.28897724, 0.61314774), -0.2765189, 49.537014, vec3(-0.66958004, 0.45035282, -0.59063095)),
    TectonicPlate(vec3(-0.8682852, -0.16594851, 0.46748477), -0.23236254, 49.537014, vec3(0.45362902, -0.646992, 0.6128801)),
    TectonicPlate(vec3(0.8529673, 0.21423429, -0.47597313), 0.23602277, 55.593468, vec3(0.48385212, -0.66657984, 0.5670612)),
    TectonicPlate(vec3(0.68821555, 0.28596404, -0.6667713), -0.22226922, 57.919964, vec3(-0.18723384, -0.81790406, -0.5440373)),
    TectonicPlate(vec3(0.67695993, 0.24926774, -0.692525), -0.22986473, 59.190933, vec3(-0.32156685, -0.7461877, -0.5829226)),
    TectonicPlate(vec3(0.68970764, 0.24122171, -0.68272644), -0.236392, 59.190933, vec3(-0.26256981, -0.7953834, -0.5462804)),
    TectonicPlate(vec3(0.8105663, 0.29784927, -0.50425005), 0.26635143, 55.593468, vec3(0.42700186, -0.8898418, 0.16078268)),
    TectonicPlate(vec3(0.6607884, 0.40715584, -0.6305417), -0.22629133, 54.17562, vec3(-0.25031647, -0.67244256, -0.69653624)),
    TectonicPlate(vec3(-0.4545163, 0.5521465, 0.698963), 0.22175944, 58.846478, vec3(0.88106287, 0.3940281, 0.2616679)),
    TectonicPlate(vec3(-0.29539505, 0.54401505, 0.7853593), 0.23929027, 58.432438, vec3(0.37006053, -0.69271606, 0.61903125)),
    TectonicPlate(vec3(-0.42060938, 0.55077577, 0.72092575), 0.23587625, 58.582535, vec3(0.86194944, -0.00533904, 0.5069661)),
    TectonicPlate(vec3(-0.43457094, 0.56824714, 0.6987441), 0.22874336, 58.846478, vec3(0.89979833, 0.24044727, 0.3640716)),
    TectonicPlate(vec3(-0.23474325, 0.66267097, 0.7111701), 0.22830667, 54.673428, vec3(-0.6909433, -0.6283583, 0.35743976)),
    TectonicPlate(vec3(-0.13862608, 0.49421793, 0.85821414), -0.22194448, 51.904488, vec3(-0.28121662, -0.8505538, 0.44438198))
);

/**
 * Advanced Terrain Erosion Filter
 *
 * @param p World or UV coordinates for evaluation.
 * @param heightAndSlope Input height (x) and derivative (yz).
 * @param fadeTarget Value to fade towards (-1 at valleys, 1 at peaks).
 * @param strength Overall erosion strength.
 * @param gullyWeight Balance between sharpening and gullies (0 to 1).
 * @param detail Detail restriction (higher = more detail on shallow slopes).
 * @param rounding Rounding parameters (x: ridge, y: crease, z: input mult, w: octave mult).
 * @param onset Controls where erosion takes effect (x: initial, y: octave, z: ridgemap initial, w: ridgemap octave).
 * @param assumedSlope Override slope for initial gully direction (x: value, y: override amount).
 * @param scale Spatial scale of the erosion.
 * @param octaves Number of gully scales to layer.
 * @param lacunarity Frequency multiplier per octave.
 * @param gain Magnitude multiplier per octave.
 * @param cellScale Size of Phacelle noise cells.
 * @param normalization Consistency of gully magnitudes.
 * @param ridgeMap Output: -1 on creases, 1 on ridges.
 * @param substrate Output: -1 for heavy erosion (valleys), 1 for deposition (plains/ridges).
 * @param debug Output: for visualization.
 * @return vec4(heightDelta, slopeDelta.xy, totalMagnitude)
 */
vec4 ErosionFilter(
	in vec2   p,
    float     texelSize,
	vec3      heightAndSlope,
	float     fadeTarget,
	float     strength,
	float     gullyWeight,
	float     detail,
	vec4      rounding,
	vec4      onset,
	vec2      assumedSlope,
	float     scale,
	int       octaves,
	float     lacunarity,
	float     gain,
	float     cellScale,
	float     normalization,
	out float ridgeMap,
	out float substrate,
	out vec2  flowDir,
	out float streamMask
) {
	strength *= (scale + 10.0);
	fadeTarget = clamp(fadeTarget, -1.0, 1.0);
	scale *= 1000.0;

	vec3  inputHeightAndSlope = heightAndSlope;
	float freq = 1.0 / (scale * cellScale);
    float fadeFreqStart = 1.0 / (4.0 * texelSize);
    float fadeFreqEnd   = 1.0 / (2.0 * texelSize);

	float slopeLength = max(length(heightAndSlope.yz), 1e-10);
	float magnitude = 0.0;
	float roundingMult = 1.0;

	float roundingForInput = mix(rounding.y, rounding.x, clamp01(fadeTarget + 0.5)) * rounding.z;
	float combiMask = ease_out(smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

	float ridgeMapCombiMask = ease_out(slopeLength * onset.z);
	float ridgeMapFadeTarget = fadeTarget;

	vec2 gullySlope = mix(heightAndSlope.yz, heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);

    float streamAccum = 0.0;
    float streamWeightSum = 0.0;

	for (int i = 0; i < octaves; i++) {
        if (freq >= fadeFreqEnd) break;

        // 1.0 below fadeFreqStart, smoothly fading to 0.0 at fadeFreqEnd
        float nyquistFade = smoothstep(fadeFreqEnd, fadeFreqStart, freq);
        float octaveStrength = strength * nyquistFade;

        vec4 phacelle = PhacelleNoise(p * freq, safe_normalize(gullySlope), cellScale, 0.25, normalization);
		phacelle.zw *= -freq;
		float sloping = abs(phacelle.y);

		gullySlope += sign(phacelle.y) * phacelle.zw * octaveStrength * gullyWeight;

		vec3 gullies = vec3(phacelle.x, phacelle.y * phacelle.zw);
		vec3 fadedGullies = mix(vec3(fadeTarget, 0.0, 0.0), gullies * gullyWeight, combiMask);
		heightAndSlope += fadedGullies * octaveStrength;
		magnitude += octaveStrength;

		fadeTarget = fadedGullies.x;

        // Gather drainage creases weighted toward macro/meso valley octaves (i = 0..3)
        // so high-frequency hillside rills do not trigger standing water.
        float crease = clamp01(-phacelle.x);
        float valleyFlatness = 1.0 - smoothstep(0.08, 0.25, length(heightAndSlope.yz));
        float catchmentWeight = nyquistFade * exp(-float(i) * 0.7) * valleyFlatness;
        streamAccum += crease * max(combiMask, 0.35) * catchmentWeight;
        streamWeightSum += catchmentWeight;

		float roundingForOctave = mix(rounding.y, rounding.x, clamp01(phacelle.x + 0.5)) * roundingMult;
		float newMask = ease_out(smooth_start(sloping * onset.y, roundingForOctave * onset.y));
		combiMask = pow_inv(combiMask, detail) * newMask;

		ridgeMapFadeTarget = mix(ridgeMapFadeTarget, gullies.x, ridgeMapCombiMask);
		float newRidgeMapMask = ease_out(sloping * onset.w);
		ridgeMapCombiMask = ridgeMapCombiMask * newRidgeMapMask;

		strength *= gain;
		freq *= lacunarity;
		roundingMult *= rounding.w;
	}

	ridgeMap = ridgeMapFadeTarget * (1.0 - ridgeMapCombiMask);
	substrate = clamp(fadeTarget, -1.0, 1.0);

    // heightAndSlope.yz points uphill; negate for steepest-descent flow and blend with
    // gullySlope so flow vectors stay continuous along flattened channel bottoms.
    vec2 bedDownhill = safe_normalize(-heightAndSlope.yz);
    vec2 channelDownhill = safe_normalize(-gullySlope);
    flowDir = safe_normalize(mix(bedDownhill, channelDownhill, 0.35));

    float rawStream = (streamWeightSum > 1e-5) ? (streamAccum / streamWeightSum) : 0.0;
    streamMask = smoothstep(0.22, 0.60, rawStream);

	vec3 heightAndSlopeDelta = heightAndSlope - inputHeightAndSlope;
	return vec4(heightAndSlopeDelta, magnitude);
}


void evaluate_tectonics_geocentric(
    vec3 p_local, float k_crumple,
    out float out_base_h, out vec3 out_grad_h,
    out float out_fault,  out vec3 out_grad_f,
    out vec3 out_vel,     out mat3 out_J_vel,
    out vec3 P_geo, out vec3 dP_dx, out vec3 dP_dz,
    out vec3 out_grad_I1)
{
    float theta = p_local.x / FAKE_PLANET_RADIUS;
    float phi   = p_local.z / FAKE_PLANET_RADIUS;
    float sin_t = sin(theta); float cos_t = cos(theta);
    float sin_p = sin(phi);   float cos_p = cos(phi);

    P_geo = vec3(cos_p * sin_t, sin_p, cos_p * cos_t);
    dP_dx = vec3( cos_p * cos_t, 0.0, -cos_p * sin_t) / FAKE_PLANET_RADIUS;
    dP_dz = vec3(-sin_p * sin_t, cos_p, -sin_p * cos_t) / FAKE_PLANET_RADIUS;

    float max_kd_macro = -1e20;
    float max_kd_crump = -1e20;
    for(int i = 0; i < plates.length(); i++) {
        float d = dot(P_geo, plates[i].seed_dir);
        max_kd_macro = max(max_kd_macro, plates[i].k * d);
        max_kd_crump = max(max_kd_crump, k_crumple * d);
    }

    float sum_w_macro = 0.0; float sum_h = 0.0;
    vec3 grad_w_macro = vec3(0.0); vec3 grad_h = vec3(0.0);

    float sum_w_crump = 0.0;  vec3 sum_v = vec3(0.0);
    float sum_w2_crump = 0.0;
    vec3 grad_w_crump = vec3(0.0); mat3 J_sum_v = mat3(0.0);
    vec3 grad_w2_crump = vec3(0.0);

    // Second-derivative accumulators for I1 = trace(J_vel), needed by uplift's gradient below.
    // dw_crump_i is a scalar multiple of S_i (dw_crump_i = k_crumple*w_crump_i*S_i), which is what
    // keeps H_grad_w_crump (the Hessian of sum_w_crump) a simple closed-form per-plate sum instead
    // of needing a separately-tracked second derivative.
    float sum_w2v_dot = 0.0;
    vec3  grad_sum_w2v_dot = vec3(0.0);
    mat3  H_grad_w_crump = mat3(0.0);

    for(int i = 0; i < plates.length(); i++) {
        vec3 S = plates[i].seed_dir;
        float d = dot(P_geo, S);

        float w_macro = exp(plates[i].k * d - max_kd_macro);
        vec3 dw_macro = plates[i].k * w_macro * S;

        sum_w_macro  += w_macro;
        sum_h        += w_macro * plates[i].height;
        grad_w_macro += dw_macro;
        grad_h       += plates[i].height * dw_macro;

        float w_crump = exp(k_crumple * d - max_kd_crump);
        vec3 dw_crump = k_crumple * w_crump * S;

        sum_w_crump  += w_crump;
        sum_v        += w_crump * plates[i].velocity;
        grad_w_crump += dw_crump;
        J_sum_v      += outerProduct(plates[i].velocity, dw_crump);

        sum_w2_crump += w_crump * w_crump;
        grad_w2_crump += 2.0 * w_crump * dw_crump;

        float velDotS = dot(plates[i].velocity, S);
        sum_w2v_dot      += w_crump * w_crump * velDotS;
        grad_sum_w2v_dot += 2.0 * w_crump * dw_crump * velDotS;
        H_grad_w_crump   += (k_crumple * k_crumple * w_crump) * outerProduct(S, S);
    }

    float inv_W_macro = 1.0 / sum_w_macro;
    out_base_h = sum_h * inv_W_macro;
    out_grad_h = (grad_h - out_base_h * grad_w_macro) * inv_W_macro;

    float inv_W_crump = 1.0 / sum_w_crump;
    float inv_W_crump2 = inv_W_crump * inv_W_crump;

    out_vel   = sum_v * inv_W_crump;
    out_J_vel = (J_sum_v - outerProduct(out_vel, grad_w_crump)) * inv_W_crump;

    // grad_I1 = d(trace(J_vel))/d(P_geo). trace(J_vel) = k_crumple*sum_w2v_dot*inv_W_crump -
    // dot(sum_v,grad_w_crump)*inv_W_crump^2 (an algebraic identity, since trace(outerProduct(a,b))
    // = dot(a,b)); differentiating that closed form once more via the product/quotient rule gives
    // the expression below. H_grad_w_crump is symmetric (built from outerProduct(S,S) terms), so
    // transpose(H_grad_w_crump)*sum_v below is interchangeable with H_grad_w_crump*sum_v.
    float B_quantity = dot(sum_v, grad_w_crump);
    vec3  grad_B = transpose(J_sum_v) * grad_w_crump + H_grad_w_crump * sum_v;

    out_grad_I1 = k_crumple * inv_W_crump * grad_sum_w2v_dot
                - grad_B * inv_W_crump2
                + (2.0 * B_quantity * inv_W_crump2 * inv_W_crump - k_crumple * sum_w2v_dot * inv_W_crump2) * grad_w_crump;

    float sum_N2_crump = sum_w2_crump * inv_W_crump2;
    float raw_fault = 1.0 - sum_N2_crump;
    float max_fault_expected = 0.55;

    float t_fault = clamp(raw_fault / max_fault_expected, 0.0, 1.0);
    out_fault = t_fault * t_fault * (3.0 - 2.0 * t_fault);

    float d_smooth_fault = 6.0 * t_fault * (1.0 - t_fault) / max_fault_expected;
    vec3 raw_grad_f = -(grad_w2_crump - 2.0 * sum_N2_crump * sum_w_crump * grad_w_crump) * inv_W_crump2;
    out_grad_f = raw_grad_f * d_smooth_fault;
}

// -----------------------------------------------------------------------------
// evaluate_terrain_analytical's regions, each split into its own function below so it can be
// read/tested/gated independently of the rest. Every *Octaves parameter is passed in explicitly
// by the orchestrator at the bottom of this file with the literal count the single monolithic
// function used to hardcode -- behavior today is unchanged; a future LOD-aware caller can lower
// a region's own octave count for distant/coarse chunks without touching that region's math.
// -----------------------------------------------------------------------------

struct WrappedTerrainCoord {
    vec3 p_local;
    bool poleFlipped;
};

// Region 0: polar and seam world-coordinate wrapping.
WrappedTerrainCoord wrapTerrainCoordinates(vec3 p_local) {
    float Z_max = 0.5 * PI * FAKE_PLANET_RADIUS;
    float X_max = PI * FAKE_PLANET_RADIUS;

    WrappedTerrainCoord result;
    result.poleFlipped = false;

    // North / South polar wrapping (Z axis past poles)
    float periodZ = 4.0 * Z_max;
    p_local.z = mod(p_local.z + 2.0 * Z_max, periodZ);
    if (p_local.z < 0.0) p_local.z += periodZ;
    p_local.z -= 2.0 * Z_max;

    if (p_local.z > Z_max) {
        p_local.z = 2.0 * Z_max - p_local.z;
        p_local.x += X_max;
        result.poleFlipped = true;
    } else if (p_local.z < -Z_max) {
        p_local.x += X_max;
        p_local.z = -2.0 * Z_max - p_local.z;
        result.poleFlipped = true;
    }

    // East / West seam wrapping (X axis)
    p_local.x = mod(p_local.x + X_max, 2.0 * X_max);
    if (p_local.x < 0.0) p_local.x += 2.0 * X_max;
    p_local.x -= X_max;

    result.p_local = p_local;
    return result;
}

struct TectonicsResult {
    float base_h;
    vec3  grad_base_h;
    float fault_mask;
    vec3  grad_fault;
    vec3  vel;
    mat3  J_vel;
    vec3  P_geo;
    vec3  dP_dx;
    vec3  dP_dz;
    float collision_intensity;
};

// Region 1+2: tectonic plates (base elevation, fault lines, plate velocity field) plus the
// collision uplift they drive. warpOctaves is the domain-warp fbm run ahead of the per-plate
// evaluation itself.
TectonicsResult evaluateTectonics(vec3 p_local, int warpOctaves) {
    mat3 J_tectonicWarp;
    vec3 tectonicWarp = cross_noise_fbm(16*p_local / FAKE_PLANET_RADIUS, warpOctaves, 0.0, J_tectonicWarp);
    vec3 p_tectonic = p_local + 20000.0 * tectonicWarp;

    TectonicsResult result;
    float k_crumple = 350.0;
    vec3  dP_dx, dP_dz, grad_I1;

    evaluate_tectonics_geocentric(p_tectonic, k_crumple,
        result.base_h, result.grad_base_h, result.fault_mask, result.grad_fault,
        result.vel, result.J_vel, result.P_geo, dP_dx, dP_dz, grad_I1);

    // dP_dx/dP_dz above are d(P_geo)/d(p_tectonic.x/z), but p_tectonic is itself a warped
    // function of p_local (20000-unit-amplitude cross-noise warp, comparable in scale to a
    // plate boundary's own width) -- correct them into d(P_geo)/d(p_local.x/z) via the warp's
    // own Jacobian so downstream slope/normal math differentiates the thing that's actually
    // being evaluated, not the pre-warp position.
    {
        float warpChain = 20000.0 * 16.0 / FAKE_PLANET_RADIUS;
        vec3  dPTectonic_dLocalX = vec3(1.0, 0.0, 0.0) + warpChain * J_tectonicWarp[0];
        vec3  dPTectonic_dLocalZ = vec3(0.0, 0.0, 1.0) + warpChain * J_tectonicWarp[2];

        result.dP_dx = dP_dx * dPTectonic_dLocalX.x + dP_dz * dPTectonic_dLocalX.z;
        result.dP_dz = dP_dx * dPTectonic_dLocalZ.x + dP_dz * dPTectonic_dLocalZ.z;
    }

    // Uplift
    mat3 S = 0.5 * (result.J_vel + transpose(result.J_vel));
    float I1 = S[0][0] + S[1][1] + S[2][2];
    result.collision_intensity = smoothstep(-0.5, -2.0, I1);
    float uplift = result.collision_intensity * 0.4;
    result.base_h += uplift;

    // d(collision_intensity)/dI1, matching GLSL smoothstep's own cubic (edge0=-0.5 > edge1=-2.0,
    // so this is decreasing in I1; zero outside the transition band, same as smoothstep's clamp).
    float t_uplift = clamp((I1 - (-0.5)) / (-2.0 - (-0.5)), 0.0, 1.0);
    float d_collision_dI1 = (t_uplift > 0.0 && t_uplift < 1.0)
        ? (6.0 * t_uplift * (1.0 - t_uplift) / (-2.0 - (-0.5)))
        : 0.0;
    result.grad_base_h += (d_collision_dI1 * 0.4) * grad_I1;

    return result;
}

struct ContinentResult {
    float continent_h;
    vec3  grad_continent_h;
    float interior_mask;
    vec3  grad_interior_mask;
    float hill_h;
    vec3  grad_hill_h;
    float t_macro; // reused by applyErosion's rounding tuning
};

// Region 3: base continent elevation plus the hill-country regional biome layer riding on top
// of it, inside/outside the fault lines.
ContinentResult evaluateContinentAndBiomeRegions(
    vec3  P_noise,
    float radius_scale,
    float phase,
    float fault_mask,
    vec3  grad_fault,
    int   continentOctaves,
    int   macroOctaves,
    int   hillsOctaves
) {
    ContinentResult result;

    vec3 grad_base_noise;
    float base_noise = dot_noise_fbm(P_noise, continentOctaves, phase+32, grad_base_noise);
    float base_noise_amp = 0.95;
    result.continent_h = base_noise * base_noise_amp;
    result.grad_continent_h = grad_base_noise * base_noise_amp * radius_scale;

    result.interior_mask = 1.0 - fault_mask;
    result.grad_interior_mask = -grad_fault;

    // Evaluate macro biome noise to separate plains from hill country
    vec3 grad_macro_noise;
    float macro_noise = dot_noise_fbm(P_noise * 0.1, macroOctaves, phase+123.0, grad_macro_noise);

    // Smoothstep maps noise range [-0.2, 0.4] to multiplier [0.0, 2.5]
    result.t_macro = clamp((macro_noise - (-0.2)) / 0.6, 0.0, 1.0);
    float hill_multiplier = result.t_macro * result.t_macro * (3.0 - 2.0 * result.t_macro) * 2.5;
    float d_hill_multiplier = 2.5 * 6.0 * result.t_macro * (1.0 - result.t_macro) / 0.6;
    vec3 grad_hill_multiplier = d_hill_multiplier * grad_macro_noise * 0.1 * radius_scale;

    vec3 grad_raw_hills;
    float raw_hills = dot_noise_fbm(P_noise, hillsOctaves, phase+89, grad_raw_hills);
    float hill_profile = pow(1.0 - abs(raw_hills), 2.0);
    vec3 grad_hill_profile = -2.0 * (1.0 - abs(raw_hills)) * sign(raw_hills) * grad_raw_hills * radius_scale;

    // Product rule for hills * interior * biome macro mask
    result.hill_h = hill_profile * result.interior_mask * hill_multiplier;
    result.grad_hill_h = (grad_hill_profile * result.interior_mask * hill_multiplier) +
                         (hill_profile * result.grad_interior_mask * hill_multiplier) +
                         (hill_profile * result.interior_mask * grad_hill_multiplier);

    return result;
}

struct MountainResult {
    float ridge_h;
    vec3  grad_ridge;
    float massive_ridge; // reused by the orchestrator's final out_ridge_map combine
    vec4  shelf;         // smax_quad_deriv(continent, hill): x = height, yzw = gradient
};

// Region 4: mountain ridges along the fault lines, smoothly merged with the continent/hill
// base from evaluateContinentAndBiomeRegions.
MountainResult evaluateMountains(
    vec3  P_noise,
    float radius_scale,
    vec3  vel,
    mat3  J_vel,
    float warp_strength,
    float phase,
    float fault_mask,
    vec3  grad_fault,
    float continent_h,
    vec3  grad_continent_h,
    float hill_h,
    vec3  grad_hill_h,
    int   ridgeOctaves
) {
    MountainResult result;

    vec3 P_warped = P_noise + vel * warp_strength;
    mat3 J_warp = mat3(radius_scale) + J_vel * warp_strength;

    vec3 grad_raw_ridge;
    float raw_ridge = dot_noise_fbm(P_warped, ridgeOctaves, phase + 42.0, grad_raw_ridge);
    vec3 grad_ridge_warped = transpose(J_warp) * grad_raw_ridge;

    float exponent = 3.0;
    float massive_ridge = pow(abs(raw_ridge), exponent);
    vec3 grad_massive_ridge = exponent * pow(abs(raw_ridge), exponent - 1.0) * sign(raw_ridge) * grad_ridge_warped;
    result.massive_ridge = massive_ridge;

    result.ridge_h = massive_ridge * fault_mask;
    result.grad_ridge = (grad_massive_ridge * fault_mask) + (massive_ridge * grad_fault);

    result.shelf = smax_quad_deriv(vec4(continent_h, grad_continent_h), vec4(hill_h, grad_hill_h), 0.25);

    return result;
}

struct HydrologyResult {
    float river_bed_h;
    vec3  grad_river_bed_h;
    float river_water_h;
    float river_profile;
    float river_mask;

    float canyon_bed_h;
    vec3  grad_canyon_bed_h;
    float canyon_water_h;
    float canyon_profile;
    float canyon_mask;

    float basin_mask;
    vec3  grad_basin_mask;
    float lake_water_level;
    vec3  grad_lake_water_level;
};

// Region 5: rivers, canyons, and lake basins with cross-sectionally flat centerline projection.
HydrologyResult evaluateHydrology(
    vec3  P_geo,
    float radius_scale,
    vec3  vel,
    mat3  J_vel,
    float warp_strength,
    float phase,
    float interior_mask,
    vec3  grad_interior_mask,
    float base_h,
    vec3  grad_base_h,
    float macro_h,
    vec3  macro_grad_geo,
    vec3  smooth_grad_geo, // grad_base_h + grad_continent_h (no high-octave hill gradients)
    int   riverOctaves,
    int   canyonOctaves,
    int   basinOctaves
) {
    HydrologyResult result;

    // Suppress rivers, canyons, and lakes on steep mountain walls
    float physical_slope  = length(smooth_grad_geo) * (2500.0 / FAKE_PLANET_RADIUS);
    float river_flatness  = 1.0 - smoothstep(0.12, 0.35, physical_slope);
    float canyon_flatness = 1.0 - smoothstep(0.18, 0.45, physical_slope);
    float basin_flatness  = (1.0 - smoothstep(0.04, 0.14, physical_slope)) * interior_mask;

    // 1. RIVERS
    vec3 grad_raw_river;
    vec3 P_river = P_geo * radius_scale * 2.0 + vel * warp_strength * 1.5;
    float raw_river = dot_noise_fbm(P_river, riverOctaves, phase + 112.0, grad_raw_river);
    vec3 grad_river_warped = transpose(mat3(radius_scale * 2.0) + J_vel * warp_strength * 1.5) * grad_raw_river;

    float river_width = 0.035;
    float t_river = clamp(abs(raw_river) / river_width, 0.0, 1.0);
    result.river_profile = 1.0 - (t_river * t_river * (3.0 - 2.0 * t_river));
    result.river_mask = interior_mask * river_flatness;

    float river_grad_sq = max(dot(grad_river_warped, grad_river_warped), 1e-8);
    vec3 delta_P_river = -(raw_river / river_grad_sq) * grad_river_warped;
    float max_river_step = 0.05 / max(radius_scale, 1e-4);
    float r_step_len = length(delta_P_river);
    if (r_step_len > max_river_step) {
        delta_P_river *= max_river_step / r_step_len;
    }

    float river_proj_delta = clamp(dot(smooth_grad_geo, delta_P_river), -0.025, 0.025);
    float river_center_h = macro_h + river_proj_delta;
    vec3 n_river = grad_river_warped * inversesqrt(river_grad_sq);
    vec3 grad_river_center_h = macro_grad_geo - n_river * dot(macro_grad_geo, n_river);

    float river_depth = 0.035;
    // Never allow a river bed to extrude above surrounding terrain on slopes
    result.river_bed_h = min(river_center_h - river_depth, macro_h - river_depth * 0.35);
    result.grad_river_bed_h = grad_river_center_h;
    result.river_water_h = result.river_bed_h + river_depth * 0.75;

    // 2. CANYONS
    vec3 grad_raw_canyon;
    float raw_canyon = dot_noise_fbm(P_geo * radius_scale * 1.5, canyonOctaves, phase + 175.0, grad_raw_canyon);
    vec3 grad_canyon_warped = grad_raw_canyon * radius_scale * 1.5;

    float canyon_width = 0.018;
    float t_canyon = clamp(abs(raw_canyon) / canyon_width, 0.0, 1.0);
    result.canyon_profile = 1.0 - (t_canyon * t_canyon * (3.0 - 2.0 * t_canyon));

    float t_elev = clamp((base_h - 0.2) / 0.3, 0.0, 1.0);
    float high_elev_mask = t_elev * t_elev * (3.0 - 2.0 * t_elev);
    result.canyon_mask = interior_mask * high_elev_mask * canyon_flatness;

    float canyon_grad_sq = max(dot(grad_canyon_warped, grad_canyon_warped), 1e-8);
    vec3 delta_P_canyon = -(raw_canyon / canyon_grad_sq) * grad_canyon_warped;
    float max_canyon_step = 0.03 / max(radius_scale, 1e-4);
    float c_step_len = length(delta_P_canyon);
    if (c_step_len > max_canyon_step) {
        delta_P_canyon *= max_canyon_step / c_step_len;
    }

    float canyon_proj_delta = clamp(dot(smooth_grad_geo, delta_P_canyon), -0.04, 0.04);
    float canyon_center_h = macro_h + canyon_proj_delta;
    vec3 n_canyon = grad_canyon_warped * inversesqrt(canyon_grad_sq);
    vec3 grad_canyon_center_h = macro_grad_geo - n_canyon * dot(macro_grad_geo, n_canyon);

    float canyon_depth = 0.09;
    result.canyon_bed_h = min(canyon_center_h - canyon_depth, macro_h - canyon_depth * 0.35);
    result.grad_canyon_bed_h = grad_canyon_center_h;
    result.canyon_water_h = result.canyon_bed_h + canyon_depth * 0.20;

    // 3. LAKE BASINS
    vec3 grad_basin_noise;
    float basin_noise = dot_noise_fbm(P_geo * radius_scale * 1.2, basinOctaves, phase + 200.0, grad_basin_noise);
    vec3 grad_basin_geo = grad_basin_noise * radius_scale * 1.2;

    float t_basin = clamp((basin_noise - 0.65) / 0.25, 0.0, 1.0);
    result.basin_mask = t_basin * t_basin * (3.0 - 2.0 * t_basin) * basin_flatness;
    float d_basin_mask = 6.0 * t_basin * (1.0 - t_basin) / 0.25;
    result.grad_basin_mask = d_basin_mask * grad_basin_geo * basin_flatness;

    float basin_grad_sq = max(dot(grad_basin_geo, grad_basin_geo), 1e-8);
    vec3 delta_P_basin = ((0.65 - basin_noise) / basin_grad_sq) * grad_basin_geo;
    float max_basin_step = 0.15 / max(radius_scale, 1e-4);
    float b_step_len = length(delta_P_basin);
    if (b_step_len > max_basin_step) {
        delta_P_basin *= max_basin_step / b_step_len;
    }

    float basin_proj_delta = clamp(dot(smooth_grad_geo, delta_P_basin), -0.035, 0.035);
    float lake_shore_h = macro_h + basin_proj_delta;
    vec3 n_basin = grad_basin_geo * inversesqrt(basin_grad_sq);
    vec3 grad_lake_shore_h = macro_grad_geo - n_basin * dot(macro_grad_geo, n_basin);

    result.lake_water_level = lake_shore_h - 0.015;
    result.grad_lake_water_level = grad_lake_shore_h;

    return result;
}

struct ErosionResult {
    float true_height;
    vec2  slope;
    float ridge_map;
    float substrate;
    vec2  flow_dir;
    float stream_mask;
};

// Region 8: derives erosion parameters from the structural fields every earlier region
// produced, then runs ErosionFilter and folds its delta back into height/slope.
// - High mountain ridges (fault_mask) & high elevation fade towards ridges (+1.0)
// - Canyons & rivers carved channels fade towards creases (-1.0)
// - Lake basins buffer towards flat sediment deposition (+0.3)
ErosionResult applyErosion(
    vec2  p_xz,
    float texelSize,
    vec3  heightAndSlope, // x = true_height pre-erosion, yz = slope pre-erosion
    float fault_mask,
    float collision_intensity,
    float river_profile,
    float river_mask,
    float canyon_profile,
    float canyon_mask,
    float basin_mask,
    float t_macro,
    vec2  assumedSlopeDir, // already normalized by the caller
    float maxHeight,
    int   erosionOctaves
) {
    float frameworkHeightPeak = maxHeight;
    float elevRatio = clamp(heightAndSlope.x / (frameworkHeightPeak * 0.6), -1.0, 1.0);
    float erosionFadeTarget = clamp(
        mix(elevRatio, 1.0, fault_mask * 0.5) - canyon_mask * 0.6 - river_profile * river_mask * 0.4 + basin_mask * 0.5,
        -1.0, 1.0
    );

    // Erosion strength increases in mountain/canyon fault regions and reduces in flat lake basins and water regions in general (?)
    float hydrology_protection = max(basin_mask * 0.7,
                                max(river_profile * river_mask,
                                    canyon_profile * canyon_mask));

    float erosion_strength = mix(0.08, 0.20, clamp01(fault_mask * 0.7 + canyon_mask * 0.3)) * (1.0 - hydrology_protection);

    // Gully carving weight is sharper in steep mountain fault zones and canyon walls
    float erosion_gully_weight = mix(0.40, 0.70, clamp01(collision_intensity + canyon_mask * 0.5));
    // Detail restriction and spatial scale adjust according to ridge features
    float erosion_detail = mix(1.2, 1.8, clamp01(collision_intensity + canyon_mask));

    float erosion_scale = mix(0.28, 0.42, fault_mask);
    float scale_arg     = 15.0 * erosion_scale; // 4.2..6.3 -> 4.2km..6.3km after scale *= 1000.0
    float scale_norm    = 21.0 / (scale_arg + 10.0);
    vec4 erosion_rounding = vec4(mix(5.0, 10.0, t_macro), mix(0.25, 0.0, t_macro), 0.1, 2.0);

    float ridgeMapVal = 0.0;
    float substrateVal = 0.0;
    float streamMask = 0.0;
    vec2 flowDir = vec2(0.0);

    vec4 erosionDelta = ErosionFilter(
        p_xz,
        texelSize,
        heightAndSlope,
        erosionFadeTarget,
        erosion_strength * scale_norm,
        20*erosion_gully_weight,
        erosion_detail,
        erosion_rounding,            // Rounding parameters (x: ridge, y: crease, z: input mult, w: octave mult)
        vec4(1.25, 1.25, 5.8, 1.5),  // effect onset (x: initial, y: octave, z: ridgemap initial, w: ridgemap octave).
        assumedSlopeDir,
        scale_arg,
        erosionOctaves,
        2.3,                         // lacunarity
        0.42,                        // Alternatively, need to clamp wavelength to 6-8M
        1.0,                        // cellScale
        0.0,                         // normalization
        ridgeMapVal,
        substrateVal,
        flowDir,
        streamMask
    );

    ErosionResult result;
    result.true_height = heightAndSlope.x + erosionDelta.x;
    result.slope = heightAndSlope.yz + erosionDelta.yz;
    result.ridge_map = ridgeMapVal;
    result.substrate = substrateVal;
    result.flow_dir = flowDir;
    result.stream_mask = streamMask;
    return result;
}

// Region 9: geological biome classification from regional lithology and tectonic/hydrological
// structure.
float classifyGeoBiome(
    vec3  P_geo,
    float phase,
    float fault_mask,
    float collision_intensity,
    float canyon_mask,
    float basin_mask,
    float base_h,
    int   geoBiomeOctaves
) {
    vec3 grad_geo_noise;
    float raw_geo_noise = dot_noise_fbm(P_geo * 4.5, geoBiomeOctaves, phase + 314.15, grad_geo_noise);
    float base_geo = clamp(raw_geo_noise * 0.5 + 0.5, 0.0, 1.0);

    float fault_factor = clamp(fault_mask * 0.8 + collision_intensity * 0.5, 0.0, 1.0);
    float canyon_factor = clamp(canyon_mask * 0.85, 0.0, 1.0);
    float basin_factor = clamp(basin_mask * 0.75 + (1.0 - fault_mask) * smoothstep(0.1, -0.2, base_h), 0.0, 1.0);

    float geo_biome = base_geo;
    geo_biome = mix(geo_biome, 0.10, basin_factor * 0.7);
    geo_biome = mix(geo_biome, 0.35, canyon_factor * 0.8);
    geo_biome = mix(geo_biome, 0.65, fault_factor * 0.75);

    return clamp(geo_biome, 0.0, 1.0);
}

// Region 10: water level in normalized height space -- rivers/canyons/lake basins each fill to
// their own fraction of their own carved depth, highest-coverage mask wins where they overlap.
// Region 10: Unscaled flat water surface elevation inside wetted corridors
float calculateLocalWaterLevel(
    float pre_erosion_height,
    float post_erosion_height,
    float stream_mask,
    HydrologyResult hydrology,
    float min_height,
    float max_height
) {
    float height_amp = max_height - min_height;
    float ground_ref = min(pre_erosion_height, post_erosion_height);

    // Keep dry water table well below ground so coarse LOD triangle interpolation never pokes through
    float water_y = ground_ref - 60.0;

    float river_cov  = hydrology.river_profile * hydrology.river_mask;
    float canyon_cov = hydrology.canyon_profile * hydrology.canyon_mask;
    float basin_cov  = hydrology.basin_mask;

    if (river_cov > 0.005) {
        float w_y = remap(hydrology.river_water_h, 0.0, 1.0, min_height, max_height);
        // Flat inside channel core (cov >= 0.35), smoothly tucked below bank as cov -> 0
        float bank_tuck = (1.0 - smoothstep(0.02, 0.35, river_cov)) * (0.045 * height_amp);
        water_y = max(water_y, w_y - bank_tuck);
    }

    if (canyon_cov > 0.005) {
        float w_y = remap(hydrology.canyon_water_h, 0.0, 1.0, min_height, max_height);
        float bank_tuck = (1.0 - smoothstep(0.02, 0.35, canyon_cov)) * (0.11 * height_amp);
        water_y = max(water_y, w_y - bank_tuck);
    }

    if (basin_cov > 0.005) {
        float w_y = remap(hydrology.lake_water_level, 0.0, 1.0, min_height, max_height);
        float shore_tuck = (1.0 - smoothstep(0.05, 0.35, basin_cov)) * (0.035 * height_amp);
        water_y = max(water_y, w_y - shore_tuck);
    }

    // Eroded tributary streams inside incised gully thalwegs (above sea level)
    if (stream_mask > 0.25 && post_erosion_height > 0.5) {
        float stream_t = smoothstep(0.25, 0.75, stream_mask);
        float stream_water_y = post_erosion_height + mix(-1.5, 0.40, stream_t);
        water_y = max(water_y, stream_water_y);
    }

    if (post_erosion_height < 0.0) {
        water_y = max(water_y, 0.0);
    }
    return water_y;
}

struct MicroDetailResult {
    float height_delta;
    vec2  slope_delta;
};

// Evaluates band-limited mesostructure (wavelengths ~24m down to ~1.5m at LOD 0).
// Uses P_geo to avoid float32 precision loss at planetary coordinates, and modulates
// character using existing substrate, ridge, and hydrology fields.
// Evaluates stationary directional phasor mesostructure (20m down to 1.25m, Nyquist-gated
// per LOD) matching the warp, orientation, and phase of terrain.frag.
// Evaluates stationary directional phasor mesostructure (20m down to 2.5m, Nyquist-gated
// per LOD) matching the warp, orientation, and phase of terrain.frag.
MicroDetailResult evaluateMicroDetail(
    vec3  worldPos,
    vec3  macroNormal,
    bool  poleFlipped,
    float texelSize,
    float ridge_map,
    float substrate,
    float water_level,
    float wetness_mask
) {
    MicroDetailResult res = MicroDetailResult(0.0, vec2(0.0));

    // Matches terrain.frag baseFrequency = 0.050 (lambda = 20m, 10m, 5m, 2.5m)
    float freq = 0.050;
    float wavelength = 1.0 / freq;
    float minWavelength = 2.0 * texelSize;
    if (wavelength <= minWavelength) {
        return res;
    }

    float slopeMag = length(vec2(macroNormal.x, macroNormal.z)) / max(macroNormal.y, 0.15);
    float steepness = smoothstep(0.06, 0.40, slopeMag);
    float depositional = smoothstep(-0.1, 0.6, substrate);
    float scouredRock = clamp01(-substrate) * smoothstep(0.04, 0.22, slopeMag);

    float rockiness = clamp01(
        max(ridge_map * 0.85, scouredRock) * 0.75 +
        steepness * 0.45 -
        depositional * (1.0 - steepness) * 0.45
    );

    float submergence = smoothstep(water_level - 1.5, water_level + 0.5, worldPos.y);
    float dryFactor = min(submergence, 1.0 - wetness_mask * 0.85);

    // Conservative amplitude in meters at lambda = 20m; halves per octave so slope stays gentle
    float amp = mix(0.025, 0.12, rockiness) * mix(0.2, 1.0, dryFactor);

    // Match terrain.frag's exact domain warp and tangent orientation
    mat3 warpNoiseGrad;
    vec3 warpNoise = cross_noise_fbm(worldPos * 0.00125, 3, 12.0, warpNoiseGrad);
    vec3 warpCurl = get_curl(warpNoiseGrad);

    float lat = clamp(worldPos.z / (FAKE_PLANET_RADIUS * 1.5707963), -1.0, 1.0);
    vec2 pWind = getPrevailingWind(lat);
    vec3 windDir3D = vec3(pWind.x, 0.0, pWind.y);
    float windLen = length(windDir3D);
    windDir3D = (windLen > 1e-4) ? (windDir3D / windLen) : vec3(1.0, 0.0, 0.0);

    vec3 referenceUp = abs(macroNormal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(referenceUp, macroNormal));
    vec3 windTangent = windDir3D - macroNormal * dot(windDir3D, macroNormal);
    windTangent = (length(windTangent) > 1e-4) ? normalize(windTangent) : T;

    vec3 waveDirRaw = windTangent + warpCurl;
    float waveDirLen = length(waveDirRaw);
    vec3 waveF = (waveDirLen > 1e-4) ? (waveDirRaw / waveDirLen) : windTangent;
    vec3 warpedPos = warpNoise + worldPos;

    const int MICRO_OCTAVES = 4; // 20m -> 10m -> 5m -> 2.5m
    for (int i = 0; i < MICRO_OCTAVES; i++) {
        wavelength = 1.0 / freq;
        if (wavelength <= minWavelength) break;

        float nyquistFade = smoothstep(minWavelength, 2.0 * minWavelength, wavelength);
        float w = amp * nyquistFade;

        vec3 grad3D;
        vec2 valPhase = phasorNoise3DGrad(warpedPos * freq, 1.0, waveF, windTangent, 0.68, 0.0, grad3D);
        float mag = length(valPhase);

        // Attenuate near destructive-interference nulls where phase and its gradient singularize
        float nullFade = smoothstep(0.05, 0.35, mag);
        if (nullFade > 0.0 && !any(isnan(grad3D.xz)) && !any(isinf(grad3D.xz))) {
            float phasorCos = (valPhase.x / max(mag, 1e-4)) * nullFade;

            // Soft-clamp grad3D against 1/r phase-singularity spikes while preserving 2*PI slope scale
            vec2 safeGrad = grad3D.xz / (1.0 + length(grad3D.xz) / 6.2831853);
            vec2 dHdxz = safeGrad * freq * w * nullFade;

            res.height_delta += phasorCos * w;
            res.slope_delta  += vec2(dHdxz.x, poleFlipped ? -dHdxz.y : dHdxz.y);
        }

        freq *= 2.0;
        amp *= 0.5;
    }

    float slopeDeltaLen = length(res.slope_delta);
    if (slopeDeltaLen > 0.30) {
        res.slope_delta *= 0.30 / slopeDeltaLen;
    }

    return res;
}


// Integrated analytical terrain evaluation with guided erosion filter -- threads the region
// functions above together in the same order/shape the original monolithic version did.
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    float lod,
    float texelSize,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map,
    out float out_substrate,
    out float out_biome
) {
    // 0. Apply polar and seam world coordinate wrapping
    WrappedTerrainCoord wrapped = wrapTerrainCoordinates(p_local);
    p_local = wrapped.p_local;

    // 1+2. Evaluate Tectonics, then the collision Uplift they drive
    TectonicsResult tectonics = evaluateTectonics(p_local, 6);

    // 3. Base Continent & Regional Biomes
    float radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
    vec3 P_noise = tectonics.P_geo * radius_scale;

    // int hillsOctaves = clamp(9 - int(round(log2(max(texelSize, 0.5)))), 4, 6);
    ContinentResult continentRegion = evaluateContinentAndBiomeRegions(
        P_noise, radius_scale, phase, tectonics.fault_mask, tectonics.grad_fault, 5, 3, 10
    );

    // 4. Mountains
    MountainResult mountains = evaluateMountains(
        P_noise, radius_scale, tectonics.vel, tectonics.J_vel, warp_strength, phase,
        tectonics.fault_mask, tectonics.grad_fault,
        continentRegion.continent_h, continentRegion.grad_continent_h,
        continentRegion.hill_h, continentRegion.grad_hill_h,
        6
    );

    // 5. Hydrology (Rivers, Canyons, Lakes)
    float macro_h = tectonics.base_h + mountains.shelf.x + mountains.ridge_h;
    vec3 macro_grad_geo = tectonics.grad_base_h + mountains.shelf.yzw + mountains.grad_ridge;

    HydrologyResult hydrology = evaluateHydrology(
        tectonics.P_geo, radius_scale, tectonics.vel, tectonics.J_vel, warp_strength, phase,
        continentRegion.interior_mask, continentRegion.grad_interior_mask,
        tectonics.base_h, tectonics.grad_base_h,
        macro_h, macro_grad_geo, tectonics.grad_base_h + continentRegion.grad_continent_h,
        4, 4, 3
    );

    // 6. Final Composition (blend toward cross-sectionally flat beds rather than subtracting)
    float river_w = hydrology.river_profile * hydrology.river_mask;
    float h_after_river = mix(macro_h, hydrology.river_bed_h, river_w);
    vec3 grad_after_river = mix(macro_grad_geo, hydrology.grad_river_bed_h, river_w);

    float canyon_w = hydrology.canyon_profile * hydrology.canyon_mask;
    float raw_final_h = mix(h_after_river, hydrology.canyon_bed_h, canyon_w);
    vec3 raw_final_grad = mix(grad_after_river, hydrology.grad_canyon_bed_h, canyon_w);

    float lake_bed_h = hydrology.lake_water_level - 0.025;
    float final_h = mix(raw_final_h, lake_bed_h, hydrology.basin_mask);
    vec3 final_grad_geo = mix(raw_final_grad, hydrology.grad_lake_water_level, hydrology.basin_mask)
        + (lake_bed_h - raw_final_h) * hydrology.grad_basin_mask;

    // 7. Resolve Un-eroded Height and Normal
    float true_height = remap(final_h, 0.0, 1.0, config.min_height, config.max_height);
    float height_amplitude = config.max_height - config.min_height;

    final_grad_geo *= height_amplitude;

    vec3 grad_local = vec3(dot(final_grad_geo, tectonics.dP_dx), 0.0, dot(final_grad_geo, tectonics.dP_dz));

    // 8. Integrated Erosion Filter Guided by Structural Generation Fields
    // Project 3D geocentric velocity into the local 2D plane
    vec2 local_vel = vec2(dot(tectonics.vel, tectonics.dP_dx), dot(tectonics.vel, tectonics.dP_dz));

    // Skew the slope vector so erosion tracks along the fold direction
    vec2 tectonic_skewed_slope = grad_local.xz + local_vel * 0.5;
    vec3 heightAndSlope = vec3(true_height, tectonic_skewed_slope);

    // 8. Erosion
    ErosionResult erosion = applyErosion(
        p_local.xz, texelSize, heightAndSlope,
        tectonics.fault_mask, tectonics.collision_intensity,
        hydrology.river_profile, hydrology.river_mask,
        hydrology.canyon_profile, hydrology.canyon_mask,
        hydrology.basin_mask, continentRegion.t_macro,
        normalize(local_vel), config.max_height,
        8
    );

    // Keep positive ridges in out_ridge_map; negative creases are now in erosion.stream_mask
    out_ridge_map = max(mountains.massive_ridge, max(0.0, erosion.ridge_map));
    out_substrate = erosion.substrate;

    // 9. Water level (computed before micro-detail so shoreline/underwater relief is suppressed)
    out_water_level = calculateLocalWaterLevel(
        true_height, erosion.true_height, erosion.stream_mask, hydrology,
        config.min_height, config.max_height
    );

    // 10. Close-up band-limited mesostructure
    vec3 macro_normal = normalize(vec3(
        -erosion.slope.x,
        1.0,
        wrapped.poleFlipped ? erosion.slope.y : -erosion.slope.y
    ));
    float wetness_mask = clamp01(max(max(river_w, canyon_w), max(hydrology.basin_mask, erosion.stream_mask)));

    MicroDetailResult micro = evaluateMicroDetail(
        vec3(p_local.x, erosion.true_height, p_local.z),
        macro_normal,
        wrapped.poleFlipped,
        texelSize,
        out_ridge_map,
        erosion.substrate,
        out_water_level,
        wetness_mask
    );

    true_height = erosion.true_height + micro.height_delta;
    vec2 final_slope = erosion.slope + micro.slope_delta;
    out_normal = normalize(vec3(-final_slope.x, 1.0, wrapped.poleFlipped ? final_slope.y : -final_slope.y));

    // 9. Derive geological biome classification from regional lithology and tectonic/hydrological structure
    out_biome = classifyGeoBiome(
        tectonics.P_geo, phase, tectonics.fault_mask, tectonics.collision_intensity,
        hydrology.canyon_mask, hydrology.basin_mask, tectonics.base_h, 3
    );

    return true_height;
}

// Overload for callers requesting out_ridge_map and out_substrate
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    float lod,
    float texelSize,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map,
    out float out_substrate
) {
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, lod, texelSize, config, out_normal, out_water_level, out_ridge_map, out_substrate, dummyBiome);
}

// Overload for callers requesting out_ridge_map
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    float lod,
    float texelSize,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map
) {
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, lod, texelSize, config, out_normal, out_water_level, out_ridge_map, dummySubstrate, dummyBiome);
}

// Overload for callers requesting out_water_level
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    float lod,
    float texelSize,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level
) {
    float dummyRidgeMap;
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, lod, texelSize, config, out_normal, out_water_level, dummyRidgeMap, dummySubstrate, dummyBiome);
}

// Overload for callers requesting only out_normal
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    float lod,
    float texelSize,
    TerrainConfig config,
    out vec3 out_normal
) {
    float dummyWaterLevel;
    float dummyRidgeMap;
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, lod, texelSize, config, out_normal, dummyWaterLevel, dummyRidgeMap, dummySubstrate, dummyBiome);
}