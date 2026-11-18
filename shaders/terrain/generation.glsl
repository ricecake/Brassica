#include "common.glsl"
#include "terrain.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "lygia/generative/noised.glsl"

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

struct PlateFlow {
    int mode;        // 0 = channel, 1 = 1-cell pond, 2 = 2-cell mutual pond
    vec2 targetSkew; // Downhill neighbor (mode 0) or pond partner (mode 2)
    float plateH;
    float waterH;
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
	out float debug
) {
	strength *= (scale + 10.0);
	fadeTarget = clamp(fadeTarget, -1.0, 1.0);

	scale *= 100.0;

	vec3  inputHeightAndSlope = heightAndSlope;
	float freq = 1.0 / (scale * cellScale);
	float slopeLength = max(length(heightAndSlope.yz), 1e-10);
	float magnitude = 0.0;
	float roundingMult = 1.0;

	float roundingForInput = mix(rounding.y, rounding.x, clamp01(fadeTarget + 0.5)) * rounding.z;
	float combiMask = ease_out(smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

	float ridgeMapCombiMask = ease_out(slopeLength * onset.z);
	float ridgeMapFadeTarget = fadeTarget;

	vec2 gullySlope = mix(heightAndSlope.yz, heightAndSlope.yz / slopeLength * assumedSlope.x, assumedSlope.y);

	for (int i = 0; i < octaves; i++) {
		vec4 phacelle = PhacelleNoise(p * freq, safe_normalize(gullySlope), cellScale, 0.25, normalization);
		phacelle.zw *= -freq;
		float sloping = abs(phacelle.y);

		gullySlope += sign(phacelle.y) * phacelle.zw * strength * gullyWeight;

		vec3 gullies = vec3(phacelle.x, phacelle.y * phacelle.zw);
		vec3 fadedGullies = mix(vec3(fadeTarget, 0.0, 0.0), gullies * gullyWeight, combiMask);
		heightAndSlope += fadedGullies * strength;
		magnitude += strength;

		fadeTarget = fadedGullies.x;

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
	debug = fadeTarget;

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

struct ErosionResult {
    float true_height;
    vec2  slope;
    float ridge_map;
    float substrate;
};

// Region 8: derives erosion parameters from the structural fields every earlier region
// produced, then runs ErosionFilter and folds its delta back into height/slope.
// - High mountain ridges (fault_mask) & high elevation fade towards ridges (+1.0)
// - Canyons & rivers carved channels fade towards creases (-1.0)
// - Lake basins buffer towards flat sediment deposition (+0.3)
ErosionResult applyErosion(
    vec2  p_xz,
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

    float erosion_scale = mix(0.12, 0.22, fault_mask);
    vec4 erosion_rounding = vec4(mix(5.0, 10.0, t_macro), mix(0.25, 0.0, t_macro), 0.1, 2.0);

    float ridgeMapVal = 0.0;
    float substrateVal = 0.0;
    float debugVal = 0.0;

    vec4 erosionDelta = ErosionFilter(
        p_xz,
        heightAndSlope,
        erosionFadeTarget,
        7*erosion_strength,
        2*erosion_gully_weight,
        erosion_detail,
        erosion_rounding,            // Rounding parameters (x: ridge, y: crease, z: input mult, w: octave mult)
        vec4(1.25, 1.25, 5.8, 1.5),  // effect onset (x: initial, y: octave, z: ridgemap initial, w: ridgemap octave).
        assumedSlopeDir,
        5*erosion_scale,
        erosionOctaves,
        2.3,                         // lacunarity
        0.52,                        // gain
        10.0,                        // cellScale
        0.5,                         // normalization
        ridgeMapVal,
        substrateVal,
        debugVal
    );

    ErosionResult result;
    result.true_height = heightAndSlope.x + erosionDelta.x;
    result.slope = heightAndSlope.yz + erosionDelta.yz;
    result.ridge_map = ridgeMapVal;
    result.substrate = substrateVal;
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

// --- Simplex-Hex Hydrology Constants & Helpers ---
const float HYDRO_F2 = 0.36602540378;
const float HYDRO_G2 = 0.21132486540;

const vec2 HYDRO_HEX_DIRS[6] = vec2[6](
    vec2( 1.0,  0.0),
    vec2( 1.0,  1.0),
    vec2( 0.0,  1.0),
    vec2(-1.0,  0.0),
    vec2(-1.0, -1.0),
    vec2( 0.0, -1.0)
);

float hydroCross2d(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

float hydroHash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

int hydroHash3(vec2 v) {
    int sum = int(round(v.x)) + int(round(v.y));
    return (sum % 3 + 3) % 3;
}

vec2 hydroSkewToCart(vec2 v_skew) {
    return v_skew - (v_skew.x + v_skew.y) * HYDRO_G2;
}

float hydroSdLine(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
    return length(pa - ba * h);
}

float hydroSmin(float a, float b, float k, out float blendFactor) {
    blendFactor = clamp(0.5 + 0.5 * (a - b) / max(k, 1e-5), 0.0, 1.0);
    return mix(a, b, blendFactor) - k * blendFactor * (1.0 - blendFactor);
}

float hydroGetRadius(vec2 v_skew, vec2 v_cart, vec2 target_cart, float lobes) {
    vec2 dir = target_cart - v_cart;
    float theta = atan(dir.y, dir.x);
    float offset = hydroHash21(v_skew) * 6.2831853;
    return 0.5 + 0.22 * sin(lobes * theta + offset);
}

vec2 hydroGetEdgePoint(vec2 vA_skew, vec2 vA, vec2 vB_skew, vec2 vB, float lobes) {
    float rA = hydroGetRadius(vA_skew, vA, vB, lobes);
    float rB = hydroGetRadius(vB_skew, vB, vA, lobes);
    float t = rA / max(rA + rB, 1e-4);
    return mix(vA, vB, t);
}

vec3 hydroGetBarycentric2D(vec2 p, vec2 a, vec2 b, vec2 c) {
    vec2 v0 = b - a;
    vec2 v1 = c - a;
    vec2 v2 = p - a;
    float det = v0.x * v1.y - v1.x * v0.y;
    if (abs(det) < 1e-5) return vec3(0.3333);
    float v = (v2.x * v1.y - v1.x * v2.y) / det;
    float w = (v0.x * v2.y - v2.x * v0.y) / det;
    return vec3(1.0 - v - w, v, w);
}

struct OrganicPlateData {
    vec2 regionVertex;
    float regionID;
    float edgeDist;
    vec2 regionCenter;
    vec2 nearestCentroid;
    vec3 barycentric;
};

OrganicPlateData getOrganicPlate(vec2 xz, float scale, float lobes) {
    vec2 p = xz * scale;
    vec2 s = p + (p.x + p.y) * HYDRO_F2;
    vec2 i = floor(s);
    vec2 f = s - i;
    vec2 i1 = (f.x > f.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);

    vec2 v0_skew = i;
    vec2 v1_skew = i + i1;
    vec2 v2_skew = i + vec2(1.0, 1.0);

    vec2 v0 = hydroSkewToCart(v0_skew);
    vec2 v1 = hydroSkewToCart(v1_skew);
    vec2 v2 = hydroSkewToCart(v2_skew);

    vec2 V_skew[3];
    vec2 V_cart[3];
    V_skew[hydroHash3(v0_skew)] = v0_skew; V_cart[hydroHash3(v0_skew)] = v0;
    V_skew[hydroHash3(v1_skew)] = v1_skew; V_cart[hydroHash3(v1_skew)] = v1;
    V_skew[hydroHash3(v2_skew)] = v2_skew; V_cart[hydroHash3(v2_skew)] = v2;

    vec2 E01 = hydroGetEdgePoint(V_skew[0], V_cart[0], V_skew[1], V_cart[1], lobes);
    vec2 E12 = hydroGetEdgePoint(V_skew[1], V_cart[1], V_skew[2], V_cart[2], lobes);
    vec2 E20 = hydroGetEdgePoint(V_skew[2], V_cart[2], V_skew[0], V_cart[0], lobes);
    vec2 center = (E01 + E12 + E20) / 3.0;

    bool inC0 = hydroCross2d(E01 - E20, p - E20) * hydroCross2d(E01 - E20, V_cart[0] - E20) >= 0.0;
    bool inC1 = hydroCross2d(E12 - E01, p - E01) * hydroCross2d(E12 - E01, V_cart[1] - E01) >= 0.0;
    bool inC2 = hydroCross2d(E20 - E12, p - E12) * hydroCross2d(E20 - E12, V_cart[2] - E12) >= 0.0;

    int tri_id = -1;
    if (inC0) tri_id = 0;
    else if (inC1) tri_id = 1;
    else if (inC2) tri_id = 2;

    vec2 pA = E01;
    vec2 pB = E12;
    vec2 pC = E20;
    vec3 innerBary = hydroGetBarycentric2D(p, E01, E12, E20);

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
        int closest_id = (innerBary.y <= innerBary.x && innerBary.y <= innerBary.z) ? 0
                       : ((innerBary.z <= innerBary.x && innerBary.z <= innerBary.y) ? 1 : 2);
        active_v_skew = V_skew[closest_id];
        active_v = V_cart[closest_id];
    }

    OrganicPlateData data;
    data.regionVertex = active_v_skew;
    data.regionID = hydroHash21(active_v_skew + 123.456);
    data.regionCenter = active_v;
    data.nearestCentroid = center;
    data.barycentric = hydroGetBarycentric2D(p, pA, pB, pC);
    data.edgeDist = min(hydroSdLine(p, center, E01),
                    min(hydroSdLine(p, center, E12),
                        hydroSdLine(p, center, E20)));
    return data;
}

// Scalar-only fast macro elevation lookup (strips Jacobians, Hessians, and analytical gradients)
float sampleMacroHeightFast(vec2 xz, float phase, float warp_strength, TerrainConfig config) {
    WrappedTerrainCoord w = wrapTerrainCoordinates(vec3(xz.x, 0.0, xz.y));
    vec3 p_local = w.p_local;

    mat3 dummyJ;
    vec3 tectonicWarp = cross_noise_fbm(16.0 * p_local / FAKE_PLANET_RADIUS, 1, 0.0, dummyJ);
    vec3 p_tectonic = p_local + 20000.0 * tectonicWarp;

    float theta = p_tectonic.x / FAKE_PLANET_RADIUS;
    float phi   = p_tectonic.z / FAKE_PLANET_RADIUS;
    float cos_p = cos(phi);
    vec3 P_geo  = vec3(cos_p * sin(theta), sin(phi), cos_p * cos(theta));

    float k_crumple = 350.0;
    float max_kd_macro = -1e20;
    float max_kd_crump = -1e20;
    for (int i = 0; i < plates.length(); i++) {
        float d = dot(P_geo, plates[i].seed_dir);
        max_kd_macro = max(max_kd_macro, plates[i].k * d);
        max_kd_crump = max(max_kd_crump, k_crumple * d);
    }

    float sum_w_macro = 0.0, sum_h = 0.0;
    float sum_w_crump = 0.0, sum_w2_crump = 0.0;
    vec3 sum_v = vec3(0.0);
    vec3 grad_w_crump = vec3(0.0);
    float trace_J_sum_v = 0.0;

    for (int i = 0; i < plates.length(); i++) {
        vec3 S = plates[i].seed_dir;
        float d = dot(P_geo, S);

        float w_macro = exp(plates[i].k * d - max_kd_macro);
        sum_w_macro += w_macro;
        sum_h += w_macro * plates[i].height;

        float w_crump = exp(k_crumple * d - max_kd_crump);
        vec3 dw_crump = k_crumple * w_crump * S;
        sum_w_crump += w_crump;
        sum_v += w_crump * plates[i].velocity;
        grad_w_crump += dw_crump;
        trace_J_sum_v += dot(plates[i].velocity, dw_crump);
        sum_w2_crump += w_crump * w_crump;
    }

    float inv_W_macro = 1.0 / sum_w_macro;
    float base_h = sum_h * inv_W_macro;

    float inv_W_crump = 1.0 / sum_w_crump;
    vec3 vel = sum_v * inv_W_crump;
    float I1 = (trace_J_sum_v - dot(vel, grad_w_crump)) * inv_W_crump;
    base_h += smoothstep(-0.5, -2.0, I1) * 0.4;

    float raw_fault = 1.0 - sum_w2_crump * inv_W_crump * inv_W_crump;
    float t_fault = clamp(raw_fault / 0.55, 0.0, 1.0);
    float fault_mask = t_fault * t_fault * (3.0 - 2.0 * t_fault);
    float interior_mask = 1.0 - fault_mask;

    float radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
    vec3 P_noise = P_geo * radius_scale;

    float continent_h = dot_noise_fbm(P_noise, 2, phase + 32.0) * 0.95;
    float macro_noise = dot_noise_fbm(P_noise * 0.1, 1, phase + 123.0);
    float t_macro = clamp((macro_noise + 0.2) / 0.6, 0.0, 1.0);
    float hill_mult = t_macro * t_macro * (3.0 - 2.0 * t_macro) * 2.5;

    float raw_hills = dot_noise_fbm(P_noise, 1, phase + 89.0);
    float hill_h = pow(1.0 - abs(raw_hills), 2.0) * interior_mask * hill_mult;

    float raw_ridge = dot_noise_fbm(P_noise + vel * warp_strength, 2, phase + 42.0);
    float ridge_h = pow(abs(raw_ridge), 3.0) * fault_mask;

    // Smooth max approximation matching smax_quad_deriv(continent_h, hill_h, 0.25)
    float h_diff = clamp(0.5 + 0.5 * (hill_h - continent_h) / 0.25, 0.0, 1.0);
    float shelf_x = mix(continent_h, hill_h, h_diff) + 0.25 * h_diff * (1.0 - h_diff);

    return base_h + shelf_x + ridge_h;
}

float getPlateElevationFast(vec2 v_skew, float scale, float phase, float warp_strength, TerrainConfig config) {
    vec2 xz = hydroSkewToCart(v_skew) / scale;
    return sampleMacroHeightFast(xz, phase, warp_strength, config) + 1e-5 * hydroHash21(v_skew);
}

void findLowestNeighborFast(
    vec2 v_skew, float scale, float phase, float warp_strength, TerrainConfig config,
    out vec2 bestN, out float bestH
) {
    bestH = 1e9;
    bestN = v_skew;
    for (int k = 0; k < 6; ++k) {
        vec2 n_skew = v_skew + HYDRO_HEX_DIRS[k];
        float h = getPlateElevationFast(n_skew, scale, phase, warp_strength, config);
        if (h < bestH) {
            bestH = h;
            bestN = n_skew;
        }
    }
}

void considerCandidate(vec2 candSkew, float candH, inout vec2 bestSkew, inout float bestH) {
    if (candH < bestH) {
        bestH = candH;
        bestSkew = candSkew;
    }
}

float getBasinEdgeDist(vec2 p, vec2 A_skew, vec2 partner_skew, bool hasPartner, float lobes) {
    vec2 A_cart = hydroSkewToCart(A_skew);
    float minDist = 1e9;

    for (int k = 0; k < 6; ++k) {
        vec2 N0_skew = A_skew + HYDRO_HEX_DIRS[k];
        vec2 N1_skew = A_skew + HYDRO_HEX_DIRS[(k + 1) % 6];
        vec2 N0_cart = hydroSkewToCart(N0_skew);
        vec2 N1_cart = hydroSkewToCart(N1_skew);

        vec2 E0  = hydroGetEdgePoint(A_skew, A_cart, N0_skew, N0_cart, lobes);
        vec2 E1  = hydroGetEdgePoint(A_skew, A_cart, N1_skew, N1_cart, lobes);
        vec2 E01 = hydroGetEdgePoint(N0_skew, N0_cart, N1_skew, N1_cart, lobes);
        vec2 sectorCenter = (E0 + E1 + E01) / 3.0;

        bool skip0 = hasPartner && all(equal(N0_skew, partner_skew));
        bool skip1 = hasPartner && all(equal(N1_skew, partner_skew));

        if (!skip0) minDist = min(minDist, hydroSdLine(p, sectorCenter, E0));
        if (!skip1) minDist = min(minDist, hydroSdLine(p, sectorCenter, E1));
    }
    return minDist;
}

vec2 getMeanderWarp(vec2 xz, float warpFreq, float warpAmp) {
    vec2 q = xz * warpFreq;
    float wx = sin(q.y + 1.3 * sin(q.x * 0.7 + 2.1)) + 0.5 * cos(q.x * 1.9 - q.y * 1.3);
    float wy = cos(q.x + 1.3 * cos(q.y * 0.7 - 1.7)) + 0.5 * sin(q.y * 1.9 + q.x * 1.3);
    return vec2(wx, wy) * (warpAmp * 0.66);
}

vec3 getSimplexWeights(vec2 f) {
    if (f.x > f.y) return vec3(1.0 - f.x, f.x - f.y, f.y);
    return vec3(1.0 - f.y, f.y - f.x, f.x);
}

float sdSignedFlowPolyline(vec2 p, vec2 A, vec2 E, vec2 B, out float tTotal, out vec2 tangent) {
    vec2 ae = E - A;
    float t1 = clamp(dot(p - A, ae) / max(dot(ae, ae), 1e-8), 0.0, 1.0);
    vec2 c1 = A + ae * t1;
    float d1 = length(p - c1);

    vec2 eb = B - E;
    float t2 = clamp(dot(p - E, eb) / max(dot(eb, eb), 1e-8), 0.0, 1.0);
    vec2 c2 = E + eb * t2;
    float d2 = length(p - c2);

    vec2 tan1 = ae / max(length(ae), 1e-6);
    vec2 tan2 = eb / max(length(eb), 1e-6);

    vec2 closest;
    if (d1 < d2) {
        tTotal = 0.5 * t1;
        closest = c1;
    } else {
        tTotal = 0.5 + 0.5 * t2;
        closest = c2;
    }
    tangent = normalize(mix(tan1, tan2, smoothstep(0.35, 0.65, tTotal)));

    vec2 normal = vec2(-tangent.y, tangent.x);
    float side = (dot(p - closest, normal) >= 0.0) ? 1.0 : -1.0;
    return side * min(d1, d2);
}

struct MeanderSample {
    float signedDist;
    float tTotal;
    float curvature;
    vec2 tangent;
    vec2 gradDir;
};

MeanderSample evalMeanderPolyline(
    vec2 p, vec2 f, float scale, vec2 A, vec2 E, vec2 B,
    float warpFreq, float warpAmp, float maxCheckDist
) {
    vec3 w = getSimplexWeights(f);
    float edgeSymmetric = 3.0 * (w.x * w.y + w.y * w.z + w.z * w.x);
    float pinMask = edgeSymmetric * edgeSymmetric;

    vec2 pW = p + pinMask * getMeanderWarp(p / scale, warpFreq, warpAmp);

    MeanderSample ms;
    float d0 = sdSignedFlowPolyline(pW, A, E, B, ms.tTotal, ms.tangent);
    ms.signedDist = d0;
    ms.curvature = 0.0;
    ms.gradDir = vec2(-ms.tangent.y, ms.tangent.x);

    // Skip 4 extra SDF evaluations when well outside the channel bank envelope
    if (abs(d0) > maxCheckDist * 1.45) {
        return ms;
    }

    float eps = 0.008;
    float dummyT;
    vec2 dummyTan;
    float dXp = sdSignedFlowPolyline(pW + vec2(eps, 0.0), A, E, B, dummyT, dummyTan);
    float dXm = sdSignedFlowPolyline(pW - vec2(eps, 0.0), A, E, B, dummyT, dummyTan);
    float dYp = sdSignedFlowPolyline(pW + vec2(0.0, eps), A, E, B, dummyT, dummyTan);
    float dYm = sdSignedFlowPolyline(pW - vec2(0.0, eps), A, E, B, dummyT, dummyTan);

    vec2 grad = vec2(dXp - dXm, dYp - dYm) / (2.0 * eps);
    float gradLen = max(length(grad), 0.25);
    float laplacian = (dXp + dXm + dYp + dYm - 4.0 * d0) / (eps * eps);

    ms.signedDist = d0 / gradLen;
    ms.curvature = clamp(laplacian / gradLen, -4.0, 4.0);
    ms.gradDir = grad / gradLen;
    return ms;
}

float shapeMeanderChannelScalar(
    float baseHeight,
    float waterH,
    float channelDepth,
    float halfWidth,
    float bankWidth,
    float signedDist,
    float curvature,
    float cutbankStrength
) {
    float absDist = abs(signedDist);
    if (absDist >= bankWidth) return baseHeight;

    float bankPolarity = clamp(signedDist * curvature * cutbankStrength, -1.0, 1.0);
    float thalwegOffset = sign(curvature) * abs(bankPolarity) * (0.55 * halfWidth);
    float distFromThalweg = abs(signedDist - thalwegOffset);

    float wallPower = mix(1.85, 0.45, 0.5 + 0.5 * bankPolarity);
    float u = clamp(distFromThalweg / max(bankWidth - abs(thalwegOffset), 1e-4), 0.0, 1.0);
    float shapedU = pow(u, wallPower);

    float localDepth = channelDepth * mix(0.65, 1.35, 0.5 + 0.5 * bankPolarity);
    float bedH = waterH - localDepth * (1.0 - smoothstep(0.0, 1.0, shapedU));

    float outerBlend = smoothstep(halfWidth, bankWidth, absDist);
    float targetH = mix(min(bedH, waterH), baseHeight, outerBlend);
    return min(baseHeight, targetH);
}

struct HydrologyResult {
    float carve_delta;
    vec2  grad_carve_xz;
    float river_profile;
    float river_mask;
    float canyon_profile;
    float canyon_mask;
    float basin_mask;
    float water_surface_h;
};

// Region 5: Optimized Single-Pass Simplex-Hex Hydrology
HydrologyResult evaluateHydrology(
    vec2  p_xz,
    float macro_h,
    float interior_mask,
    float base_h,
    float phase,
    float warp_strength,
    TerrainConfig config
) {
    HydrologyResult result;
    result.carve_delta = 0.0;
    result.grad_carve_xz = vec2(0.0);
    result.river_profile = 0.0;
    result.river_mask = 0.0;
    result.canyon_profile = 0.0;
    result.canyon_mask = 0.0;
    result.basin_mask = 0.0;
    result.water_surface_h = -1e5;

    // 1. Immediate early exit for ocean floor and major tectonic fault lines
    float landMask = smoothstep(-0.02, 0.03, macro_h) * interior_mask;
    if (landMask <= 0.001) {
        return result;
    }

    float plate_scale = config.spatial_scale * 16.0;
    float lobes = 3.0;

    vec2 p = p_xz * plate_scale;
    vec2 s = p + (p.x + p.y) * HYDRO_F2;
    vec2 i = floor(s);
    vec2 f = s - i;
    bool swapXY = (f.x <= f.y);
    vec2 i1 = swapXY ? vec2(0.0, 1.0) : vec2(1.0, 0.0);

    vec2 V0 = i;
    vec2 V1 = i + i1;
    vec2 V2 = i + vec2(1.0, 1.0);

    // 2. Shared 12-vertex simplex neighborhood evaluation (all 3 triangle vertices + 9 unique ring neighbors)
    float hV0 = getPlateElevationFast(V0, plate_scale, phase, warp_strength, config);
    float hV1 = getPlateElevationFast(V1, plate_scale, phase, warp_strength, config);
    float hV2 = getPlateElevationFast(V2, plate_scale, phase, warp_strength, config);

    vec2 N0 = i + (swapXY ? vec2( 1.0,  0.0) : vec2( 0.0,  1.0));
    vec2 N1 = i + (swapXY ? vec2( 0.0, -1.0) : vec2(-1.0,  0.0));
    vec2 N2 = i + vec2(-1.0, -1.0);
    vec2 N3 = i + (swapXY ? vec2(-1.0,  0.0) : vec2( 0.0, -1.0));
    vec2 N4 = i + (swapXY ? vec2( 0.0,  2.0) : vec2( 2.0,  0.0));
    vec2 N5 = i + (swapXY ? vec2( 1.0,  2.0) : vec2( 2.0,  1.0));
    vec2 N6 = i + (swapXY ? vec2(-1.0,  1.0) : vec2( 1.0, -1.0));
    vec2 N7 = i + vec2( 2.0,  2.0);
    vec2 N8 = i + (swapXY ? vec2( 2.0,  1.0) : vec2( 1.0,  2.0));

    float hN0 = getPlateElevationFast(N0, plate_scale, phase, warp_strength, config);
    float hN1 = getPlateElevationFast(N1, plate_scale, phase, warp_strength, config);
    float hN2 = getPlateElevationFast(N2, plate_scale, phase, warp_strength, config);
    float hN3 = getPlateElevationFast(N3, plate_scale, phase, warp_strength, config);
    float hN4 = getPlateElevationFast(N4, plate_scale, phase, warp_strength, config);
    float hN5 = getPlateElevationFast(N5, plate_scale, phase, warp_strength, config);
    float hN6 = getPlateElevationFast(N6, plate_scale, phase, warp_strength, config);
    float hN7 = getPlateElevationFast(N7, plate_scale, phase, warp_strength, config);
    float hN8 = getPlateElevationFast(N8, plate_scale, phase, warp_strength, config);

    // Resolve steepest-descent neighbor & zero-cost local concavity width for V0, V1, V2
    vec2 triSkew[3] = vec2[3](V0, V1, V2);
    float triH[3]   = float[3](hV0, hV1, hV2);
    vec2 bestN[3];
    float bestNH[3];
    float triHalfW[3];

    // V0 neighbors: V1, V2, N0, N1, N2, N3
    bestN[0] = V1; bestNH[0] = hV1;
    considerCandidate(V2, hV2, bestN[0], bestNH[0]);
    considerCandidate(N0, hN0, bestN[0], bestNH[0]);
    considerCandidate(N1, hN1, bestN[0], bestNH[0]);
    considerCandidate(N2, hN2, bestN[0], bestNH[0]);
    considerCandidate(N3, hN3, bestN[0], bestNH[0]);
    float mean0 = (hV1 + hV2 + hN0 + hN1 + hN2 + hN3) * (1.0 / 6.0);
    float up0 = 1.0 + 0.45 * (step(hV0, hV1) + step(hV0, hV2) + step(hV0, hN0) + step(hV0, hN1) + step(hV0, hN2) + step(hV0, hN3));
    triHalfW[0] = 0.045 + 0.030 * (pow(up0 + clamp((mean0 - hV0) * 14.0, 0.0, 4.0), 0.45) - 1.0);

    // V1 neighbors: N4, N5, V2, V0, N3, N6
    bestN[1] = N4; bestNH[1] = hN4;
    considerCandidate(N5, hN5, bestN[1], bestNH[1]);
    considerCandidate(V2, hV2, bestN[1], bestNH[1]);
    considerCandidate(V0, hV0, bestN[1], bestNH[1]);
    considerCandidate(N3, hN3, bestN[1], bestNH[1]);
    considerCandidate(N6, hN6, bestN[1], bestNH[1]);
    float mean1 = (hN4 + hN5 + hV2 + hV0 + hN3 + hN6) * (1.0 / 6.0);
    float up1 = 1.0 + 0.45 * (step(hV1, hN4) + step(hV1, hN5) + step(hV1, hV2) + step(hV1, hV0) + step(hV1, hN3) + step(hV1, hN6));
    triHalfW[1] = 0.045 + 0.030 * (pow(up1 + clamp((mean1 - hV1) * 14.0, 0.0, 4.0), 0.45) - 1.0);

    // V2 neighbors: N5, N7, N8, N0, V0, V1
    bestN[2] = N5; bestNH[2] = hN5;
    considerCandidate(N7, hN7, bestN[2], bestNH[2]);
    considerCandidate(N8, hN8, bestN[2], bestNH[2]);
    considerCandidate(N0, hN0, bestN[2], bestNH[2]);
    considerCandidate(V0, hV0, bestN[2], bestNH[2]);
    considerCandidate(V1, hV1, bestN[2], bestNH[2]);
    float mean2 = (hN5 + hN7 + hN8 + hN0 + hV0 + hV1) * (1.0 / 6.0);
    float up2 = 1.0 + 0.45 * (step(hV2, hN5) + step(hV2, hN7) + step(hV2, hN8) + step(hV2, hN0) + step(hV2, hV0) + step(hV2, hV1));
    triHalfW[2] = 0.045 + 0.030 * (pow(up2 + clamp((mean2 - hV2) * 14.0, 0.0, 4.0), 0.45) - 1.0);

    // 3. Evaluate active channel segments (only when within rough bounding distance)
    float canyonBlend = smoothstep(0.25, 0.55, base_h);
    float channelDepth = mix(0.045, 0.135, canyonBlend);
    float pondDepth = 0.055;
    float warpFreq = 3.2 * plate_scale;
    float warpAmp = mix(0.14, 0.06, canyonBlend);
    float cutbankStrength = mix(0.42, 0.15, canyonBlend);

    float hResult = macro_h;
    vec2 gradCarveXZ = vec2(0.0);
    float maxChannelCov = 0.0;
    float waterSurface = -1e5;

    for (int idx = 0; idx < 3; ++idx) {
        if (bestNH[idx] >= triH[idx]) continue; // 1-cell pond, no outflow segment

        vec2 A_skew = triSkew[idx];
        vec2 B_skew = bestN[idx];
        vec2 A_cart = hydroSkewToCart(A_skew);
        vec2 B_cart = hydroSkewToCart(B_skew);
        vec2 E_cart = hydroGetEdgePoint(A_skew, A_cart, B_skew, B_cart, lobes);

        float maxPossibleHalfW = triHalfW[idx] + 0.04;
        float maxPossibleBankW = maxPossibleHalfW * 2.35;
        float roughDist = min(hydroSdLine(p, A_cart, E_cart), hydroSdLine(p, E_cart, B_cart));
        if (roughDist > maxPossibleBankW + warpAmp + 0.04) continue;

        // Check 2-hop target C from B only for segments close to this pixel
        vec2 C_skew;
        float hC;
        findLowestNeighborFast(B_skew, plate_scale, phase, warp_strength, config, C_skew, hC);
        if (all(equal(C_skew, A_skew))) continue; // 2-cell mutual pond handled in basin pass

        MeanderSample ms = evalMeanderPolyline(
            p, f, plate_scale, A_cart, E_cart, B_cart, warpFreq, warpAmp, maxPossibleBankW
        );

        float halfWidth = triHalfW[idx] * mix(1.0, 0.65, canyonBlend);
        float bankWidth = halfWidth * 2.35;

        if (abs(ms.signedDist) < bankWidth) {
            // If B is a 1-cell pond, clamp end elevation to B's spill level
            float endH = (hC >= bestNH[idx]) ? hC : bestNH[idx];
            float waterH = mix(triH[idx], endH, ms.tTotal);

            // Clamp water surface below the lowest bank outside the meander envelope
            float maxLateralReach = (bankWidth + warpAmp) / plate_scale;
            vec2 perpWorld = vec2(-ms.tangent.y, ms.tangent.x) * maxLateralReach;
            float hLeft  = sampleMacroHeightFast(p_xz + perpWorld, phase, warp_strength, config);
            float hRight = sampleMacroHeightFast(p_xz - perpWorld, phase, warp_strength, config);
            waterH = min(waterH, min(hLeft, hRight) - 0.002);

            float carvedH = shapeMeanderChannelScalar(
                macro_h, waterH, channelDepth, halfWidth, bankWidth, ms.signedDist, ms.curvature, cutbankStrength
            );
            float blendedH = mix(macro_h, carvedH, landMask);

            // 1D cross-section derivative * SDF spatial gradient gives 2D bank normal without extra passes
            float epsD = 0.004;
            float carvedH_p = shapeMeanderChannelScalar(
                macro_h, waterH, channelDepth, halfWidth, bankWidth, ms.signedDist + epsD, ms.curvature, cutbankStrength
            );
            float dh_dd = ((mix(macro_h, carvedH_p, landMask)) - blendedH) / epsD;
            vec2 segGradXZ = (dh_dd * plate_scale) * ms.gradDir;

            float sminBlend;
            hResult = hydroSmin(hResult, blendedH, 0.012, sminBlend);
            gradCarveXZ = mix(gradCarveXZ, segGradXZ, sminBlend);

            float cov = (1.0 - smoothstep(halfWidth * 0.5, bankWidth, abs(ms.signedDist))) * landMask;
            maxChannelCov = max(maxChannelCov, cov);

            float effectiveWaterH = mix(waterH, waterH - channelDepth * 0.65, canyonBlend);
            if (abs(ms.signedDist) < halfWidth * 1.05 && effectiveWaterH > hResult) {
                waterSurface = max(waterSurface, effectiveWaterH);
            }
        }
    }

    // 4. Pond basin carving for 1-cell and 2-cell local minima
    OrganicPlateData plate = getOrganicPlate(p_xz, plate_scale, lobes);
    int activeIdx = 0;
    if (all(equal(plate.regionVertex, triSkew[1]))) activeIdx = 1;
    else if (all(equal(plate.regionVertex, triSkew[2]))) activeIdx = 2;

    vec2 activeSkew = triSkew[activeIdx];
    float activeH = triH[activeIdx];
    vec2 activeTarget = bestN[activeIdx];
    float activeTargetH = bestNH[activeIdx];

    int pondMode = 0;
    float pondSpillH = activeTargetH;
    if (activeTargetH >= activeH) {
        pondMode = 1;
    } else {
        vec2 partnerTarget;
        float partnerTargetH;
        findLowestNeighborFast(activeTarget, plate_scale, phase, warp_strength, config, partnerTarget, partnerTargetH);
        if (all(equal(partnerTarget, activeSkew))) {
            pondMode = 2;
            float spillH = 1e9;
            for (int k = 0; k < 6; ++k) {
                vec2 nA = activeSkew + HYDRO_HEX_DIRS[k];
                if (!all(equal(nA, activeTarget))) {
                    spillH = min(spillH, getPlateElevationFast(nA, plate_scale, phase, warp_strength, config));
                }
                vec2 nB = activeTarget + HYDRO_HEX_DIRS[k];
                if (!all(equal(nB, activeSkew))) {
                    spillH = min(spillH, getPlateElevationFast(nB, plate_scale, phase, warp_strength, config));
                }
            }
            pondSpillH = spillH;
        }
    }

    if (pondMode != 0) {
        bool isTwoCell = (pondMode == 2);
        float edgeDist = getBasinEdgeDist(p, activeSkew, activeTarget, isTwoCell, lobes);
        float shoreBlend = smoothstep(0.0, 0.28, edgeDist) * landMask;
        float basinBedH = pondSpillH - pondDepth * shoreBlend;
        float carvedPondH = mix(macro_h, basinBedH, shoreBlend);

        if (carvedPondH < hResult) {
            float epsP = 0.015;
            float edgeDistX = getBasinEdgeDist(p + vec2(epsP, 0.0), activeSkew, activeTarget, isTwoCell, lobes);
            float edgeDistZ = getBasinEdgeDist(p + vec2(0.0, epsP), activeSkew, activeTarget, isTwoCell, lobes);
            float sbX = smoothstep(0.0, 0.28, edgeDistX) * landMask;
            float sbZ = smoothstep(0.0, 0.28, edgeDistZ) * landMask;
            float hPx = mix(macro_h, pondSpillH - pondDepth * sbX, sbX);
            float hPz = mix(macro_h, pondSpillH - pondDepth * sbZ, sbZ);

            hResult = carvedPondH;
            gradCarveXZ = vec2(hPx - carvedPondH, hPz - carvedPondH) * (plate_scale / epsP);
        }
        result.basin_mask = shoreBlend;

        if (pondSpillH > hResult) {
            waterSurface = max(waterSurface, pondSpillH);
        }
    }

    result.carve_delta = hResult - macro_h;
    result.grad_carve_xz = gradCarveXZ;
    result.river_profile = maxChannelCov * (1.0 - canyonBlend);
    result.river_mask = interior_mask * (1.0 - canyonBlend);
    result.canyon_profile = maxChannelCov * canyonBlend;
    result.canyon_mask = interior_mask * canyonBlend;
    result.water_surface_h = waterSurface;
    return result;
}

// Region 10: Remap flat water_surface_h into world height where wetted
float calculateLocalWaterLevel(
    float final_h,
    float water_surface_h,
    float river_profile,
    float river_mask,
    float canyon_profile,
    float canyon_mask,
    float basin_mask,
    float min_height,
    float max_height
) {
    float has_water_mask = max(smoothstep(0.05, 0.35, basin_mask),
                           max(smoothstep(0.05, 0.25, river_profile * river_mask),
                               smoothstep(0.05, 0.20, canyon_profile * canyon_mask)));

    if (water_surface_h <= final_h || has_water_mask <= 0.001) {
        return min_height - 1000.0;
    }
    return mix(min_height - 1000.0, remap(water_surface_h, 0.0, 1.0, min_height, max_height), has_water_mask);
}


// Integrated analytical terrain evaluation with guided erosion filter -- threads the region
// functions above together in the same order/shape the original monolithic version did.
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
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

    ContinentResult continentRegion = evaluateContinentAndBiomeRegions(
        P_noise, radius_scale, phase, tectonics.fault_mask, tectonics.grad_fault, 5, 3, 4
    );

    // 4. Mountains
    MountainResult mountains = evaluateMountains(
        P_noise, radius_scale, tectonics.vel, tectonics.J_vel, warp_strength, phase,
        tectonics.fault_mask, tectonics.grad_fault,
        continentRegion.continent_h, continentRegion.grad_continent_h,
        continentRegion.hill_h, continentRegion.grad_hill_h,
        6
    );

// 5. Simplex-Hex Hydrology (Channels, Canyons, 1/2-Cell Lake Basins)
    float macro_h = tectonics.base_h + mountains.shelf.x + mountains.ridge_h;
    vec3 macro_grad_geo = tectonics.grad_base_h + mountains.shelf.yzw + mountains.grad_ridge;

    HydrologyResult hydrology = evaluateHydrology(
        p_local.xz, macro_h,
        continentRegion.interior_mask, tectonics.base_h,
        phase, warp_strength, config
    );

    // 6. Final Composition (macro terrain + level-banked hydrology carve)
    float final_h = macro_h + hydrology.carve_delta;

    // 7. Resolve Un-eroded Height and Normal
    float true_height = remap(final_h, 0.0, 1.0, config.min_height, config.max_height);
    float height_amplitude = config.max_height - config.min_height;

    vec3 grad_local = vec3(
        dot(macro_grad_geo, tectonics.dP_dx),
        0.0,
        dot(macro_grad_geo, tectonics.dP_dz)
    ) * height_amplitude;
    grad_local.xz += hydrology.grad_carve_xz * height_amplitude;

    // 8. Integrated Erosion Filter Guided by Structural Generation Fields
    vec2 local_vel = vec2(dot(tectonics.vel, tectonics.dP_dx), dot(tectonics.vel, tectonics.dP_dz));
    vec2 tectonic_skewed_slope = grad_local.xz + local_vel * 0.5;
    vec3 heightAndSlope = vec3(true_height, tectonic_skewed_slope);

    ErosionResult erosion = applyErosion(
        p_local.xz, heightAndSlope,
        tectonics.fault_mask, tectonics.collision_intensity,
        hydrology.river_profile, hydrology.river_mask,
        hydrology.canyon_profile, hydrology.canyon_mask,
        hydrology.basin_mask, continentRegion.t_macro,
        normalize(local_vel), config.max_height,
        8
    );

    true_height = erosion.true_height;
    out_normal = normalize(vec3(-erosion.slope.x, 1.0, wrapped.poleFlipped ? erosion.slope.y : -erosion.slope.y));
    out_ridge_map = max(mountains.massive_ridge, erosion.ridge_map);
    out_substrate = erosion.substrate;

    // 9. Geological Biome Classification
    out_biome = classifyGeoBiome(
        tectonics.P_geo, phase, tectonics.fault_mask, tectonics.collision_intensity,
        hydrology.canyon_mask, hydrology.basin_mask, tectonics.base_h, 3
    );

    // 10. Flat Cross-Sectional Water Level
    out_water_level = calculateLocalWaterLevel(
        final_h, hydrology.water_surface_h,
        hydrology.river_profile, hydrology.river_mask,
        hydrology.canyon_profile, hydrology.canyon_mask,
        hydrology.basin_mask,
        config.min_height, config.max_height
    );

    return true_height;
}

// Overload for callers requesting out_ridge_map and out_substrate
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map,
    out float out_substrate
) {
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, out_water_level, out_ridge_map, out_substrate, dummyBiome);
}

// Overload for callers requesting out_ridge_map
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map
) {
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, out_water_level, out_ridge_map, dummySubstrate, dummyBiome);
}

// Overload for callers requesting out_water_level
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level
) {
    float dummyRidgeMap;
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, out_water_level, dummyRidgeMap, dummySubstrate, dummyBiome);
}

// Overload for callers requesting only out_normal
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    TerrainConfig config,
    out vec3 out_normal
) {
    float dummyWaterLevel;
    float dummyRidgeMap;
    float dummySubstrate;
    float dummyBiome;
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, dummyWaterLevel, dummyRidgeMap, dummySubstrate, dummyBiome);
}