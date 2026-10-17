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

// Integrated analytical terrain evaluation with guided erosion filter
float evaluate_terrain_analytical(
    vec3 p_local,
    float phase,
    float warp_strength,
    TerrainConfig config,
    out vec3 out_normal,
    out float out_water_level,
    out float out_ridge_map
) {
    // 0. Apply polar and seam world coordinate wrapping
    float Z_max = 0.5 * PI * FAKE_PLANET_RADIUS;
    float X_max = PI * FAKE_PLANET_RADIUS;

    // East / West seam wrapping (X axis)
    p_local.x = mod(p_local.x + X_max, 2.0 * X_max);
    if (p_local.x < 0.0) p_local.x += 2.0 * X_max;
    p_local.x -= X_max;

    // North / South polar wrapping (Z axis past poles)
    bool poleFlipped = false;
    float periodZ = 4.0 * Z_max;
    p_local.z = mod(p_local.z + 2.0 * Z_max, periodZ);
    if (p_local.z < 0.0) p_local.z += periodZ;
    p_local.z -= 2.0 * Z_max;

    if (p_local.z > Z_max) {
        p_local.z = 2.0 * Z_max - p_local.z;
        poleFlipped = true;
    } else if (p_local.z < -Z_max) {
        p_local.z = -2.0 * Z_max - p_local.z;
        poleFlipped = true;
    }

    mat3 J_tectonicWarp;
    vec3 tectonicWarp = cross_noise_fbm(16*p_local / FAKE_PLANET_RADIUS, 6, 0.0, J_tectonicWarp);
    vec3 p_tectonic = p_local + 20000.0 * tectonicWarp;

    // 1. Evaluate Tectonics
    float base_h, fault_mask;
    vec3 grad_base_h, grad_fault, vel, P_geo, dP_dx, dP_dz, grad_I1;
    mat3 J_vel;
    float k_crumple = 350.0;

    evaluate_tectonics_geocentric(p_tectonic, k_crumple,
        base_h, grad_base_h, fault_mask, grad_fault,
        vel, J_vel, P_geo, dP_dx, dP_dz, grad_I1);

    // dP_dx/dP_dz above are d(P_geo)/d(p_tectonic.x/z), but p_tectonic is itself a warped
    // function of p_local (20000-unit-amplitude cross-noise warp, comparable in scale to a
    // plate boundary's own width) -- correct them into d(P_geo)/d(p_local.x/z) via the warp's
    // own Jacobian so downstream slope/normal math differentiates the thing that's actually
    // being evaluated, not the pre-warp position.
    {
        float warpChain = 20000.0 * 16.0 / FAKE_PLANET_RADIUS;
        vec3  dPTectonic_dLocalX = vec3(1.0, 0.0, 0.0) + warpChain * J_tectonicWarp[0];
        vec3  dPTectonic_dLocalZ = vec3(0.0, 0.0, 1.0) + warpChain * J_tectonicWarp[2];

        vec3 dP_dx_corrected = dP_dx * dPTectonic_dLocalX.x + dP_dz * dPTectonic_dLocalX.z;
        vec3 dP_dz_corrected = dP_dx * dPTectonic_dLocalZ.x + dP_dz * dPTectonic_dLocalZ.z;
        dP_dx = dP_dx_corrected;
        dP_dz = dP_dz_corrected;
    }

    // 2. Uplift
    mat3 S = 0.5 * (J_vel + transpose(J_vel));
    float I1 = S[0][0] + S[1][1] + S[2][2];
    float collision_intensity = smoothstep(-0.5, -2.0, I1);
    float uplift = collision_intensity * 0.4;
    base_h += uplift;

    // d(collision_intensity)/dI1, matching GLSL smoothstep's own cubic (edge0=-0.5 > edge1=-2.0,
    // so this is decreasing in I1; zero outside the transition band, same as smoothstep's clamp).
    float t_uplift = clamp((I1 - (-0.5)) / (-2.0 - (-0.5)), 0.0, 1.0);
    float d_collision_dI1 = (t_uplift > 0.0 && t_uplift < 1.0)
        ? (6.0 * t_uplift * (1.0 - t_uplift) / (-2.0 - (-0.5)))
        : 0.0;
    grad_base_h += (d_collision_dI1 * 0.4) * grad_I1;

    // 3. Base Continent & Regional Biomes
    float radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
    vec3 P_noise = P_geo * radius_scale;

    vec3 grad_base_noise;
    float base_noise = dot_noise_fbm(P_noise, 5, phase+32, grad_base_noise);
    float base_noise_amp = 0.95;
    float continent_h = base_noise * base_noise_amp;
    vec3 grad_continent_h = grad_base_noise * base_noise_amp * radius_scale;

    float interior_mask = 1.0 - fault_mask;
    vec3 grad_interior_mask = -grad_fault;

    // Evaluate macro biome noise to separate plains from hill country
    vec3 grad_macro_noise;
    float macro_noise = dot_noise_fbm(P_noise * 0.1, 3, phase+123.0, grad_macro_noise);

    // Smoothstep maps noise range [-0.2, 0.4] to multiplier [0.0, 2.5]
    float t_macro = clamp((macro_noise - (-0.2)) / 0.6, 0.0, 1.0);
    float hill_multiplier = t_macro * t_macro * (3.0 - 2.0 * t_macro) * 2.5;
    float d_hill_multiplier = 2.5 * 6.0 * t_macro * (1.0 - t_macro) / 0.6;
    vec3 grad_hill_multiplier = d_hill_multiplier * grad_macro_noise * 0.1 * radius_scale;

    vec3 grad_raw_hills;
    float raw_hills = dot_noise_fbm(P_noise, 4, phase+89, grad_raw_hills);
    float hill_profile = pow(1.0 - abs(raw_hills), 2.0);
    vec3 grad_hill_profile = -2.0 * (1.0 - abs(raw_hills)) * sign(raw_hills) * grad_raw_hills * radius_scale;

    // Product rule for hills * interior * biome macro mask
    float hill_h = hill_profile * interior_mask * hill_multiplier;
    vec3 grad_hill_h = (grad_hill_profile * interior_mask * hill_multiplier) +
                       (hill_profile * grad_interior_mask * hill_multiplier) +
                       (hill_profile * interior_mask * grad_hill_multiplier);

    // 4. Mountains
    vec3 P_warped = P_noise + vel * warp_strength;
    mat3 J_warp = mat3(radius_scale) + J_vel * warp_strength;

    vec3 grad_raw_ridge;
    float raw_ridge = dot_noise_fbm(P_warped, 6, phase + 42.0, grad_raw_ridge);
    vec3 grad_ridge_warped = transpose(J_warp) * grad_raw_ridge;

    float exponent = 3.0;
    float massive_ridge = pow(abs(raw_ridge), exponent);
    vec3 grad_massive_ridge = exponent * pow(abs(raw_ridge), exponent - 1.0) * sign(raw_ridge) * grad_ridge_warped;

    float ridge_h = massive_ridge * fault_mask;
    vec3 grad_ridge = (grad_massive_ridge * fault_mask) + (massive_ridge * grad_fault);

    vec4 raw_smin_shelf = smax_quad_deriv(vec4(continent_h, grad_continent_h), vec4(hill_h, grad_hill_h), 0.25);

    // 5. Hydrology & Erosion (Rivers, Canyons, Lakes)

    // RIVERS (Inverted zero-crossings warped by velocity)
    vec3 grad_raw_river;
    vec3 P_river = P_geo * radius_scale * 2.0 + vel * warp_strength * 1.5;
    float raw_river = dot_noise_fbm(P_river, 4, phase + 112.0, grad_raw_river);
    vec3 grad_river_warped = transpose(mat3(radius_scale * 2.0) + J_vel * warp_strength * 1.5) * grad_raw_river;

    float river_width = 0.035;
    float t_river = clamp(abs(raw_river) / river_width, 0.0, 1.0);
    float river_profile = 1.0 - (t_river * t_river * (3.0 - 2.0 * t_river));
    float d_river_profile = -6.0 * t_river * (1.0 - t_river) / river_width;
    vec3 grad_river_profile = d_river_profile * sign(raw_river) * grad_river_warped;

    float river_mask = interior_mask; // Keep rivers out of the steepest fault lines
    float river_depth = 0.06;
    float river_carve = river_profile * river_mask * river_depth;
    vec3 grad_river_carve = river_depth * (grad_river_profile * river_mask + river_profile * grad_interior_mask);

    // CANYONS (Inverted zero-crossings masked to high elevation)
    vec3 grad_raw_canyon;
    float raw_canyon = dot_noise_fbm(P_geo * radius_scale * 1.5, 4, phase + 175.0, grad_raw_canyon);
    vec3 grad_canyon_warped = grad_raw_canyon * radius_scale * 1.5;

    float canyon_width = 0.015;
    float t_canyon = clamp(abs(raw_canyon) / canyon_width, 0.0, 1.0);
    float canyon_profile = 1.0 - (t_canyon * t_canyon * (3.0 - 2.0 * t_canyon));
    float d_canyon_profile = -6.0 * t_canyon * (1.0 - t_canyon) / canyon_width;
    vec3 grad_canyon_profile = d_canyon_profile * sign(raw_canyon) * grad_canyon_warped;

    float t_elev = clamp((base_h - 0.2) / 0.3, 0.0, 1.0);
    float high_elev_mask = t_elev * t_elev * (3.0 - 2.0 * t_elev);
    float d_high_elev_mask = 6.0 * t_elev * (1.0 - t_elev) / 0.3;
    vec3 grad_high_elev_mask = d_high_elev_mask * grad_base_h;

    float canyon_mask = interior_mask * high_elev_mask;
    vec3 grad_canyon_mask = grad_interior_mask * high_elev_mask + interior_mask * grad_high_elev_mask;

    float canyon_depth = 0.15;
    float canyon_carve = canyon_profile * canyon_mask * canyon_depth;
    vec3 grad_canyon_carve = canyon_depth * (grad_canyon_profile * canyon_mask + canyon_profile * grad_canyon_mask);

    // LAKE BASINS (Cellular-like structural flattening)
    vec3 grad_basin_noise;
    float basin_noise = dot_noise_fbm(P_geo * radius_scale * 1.2, 3, phase + 200.0, grad_basin_noise);

    float t_basin = clamp((basin_noise - 0.65) / 0.25, 0.0, 1.0);
    float basin_mask = t_basin * t_basin * (3.0 - 2.0 * t_basin);
    float d_basin_mask = 6.0 * t_basin * (1.0 - t_basin) / 0.25;
    vec3 grad_basin_mask = d_basin_mask * grad_basin_noise * radius_scale * 1.2;

    float lake_water_level = base_h - 0.02;
    vec3 grad_lake_water_level = grad_base_h;

    // 6. Final Composition
    float raw_final_h = base_h + raw_smin_shelf.x + ridge_h - river_carve - canyon_carve;
    vec3 raw_final_grad = grad_base_h + raw_smin_shelf.yzw + grad_ridge - grad_river_carve - grad_canyon_carve;

    // Apply the lake basin flattening via mix
    float final_h = mix(raw_final_h, lake_water_level, basin_mask);
    vec3 final_grad_geo = mix(raw_final_grad, grad_lake_water_level, basin_mask) + (lake_water_level - raw_final_h) * grad_basin_mask;

    // 7. Resolve Un-eroded Height and Normal
    float true_height = remap(final_h, 0.0, 1.0, config.min_height, config.max_height);
    float height_amplitude = config.max_height - config.min_height;

    final_grad_geo *= height_amplitude;

    vec3 grad_local = vec3(dot(final_grad_geo, dP_dx), 0.0, dot(final_grad_geo, dP_dz));

    // 8. Integrated Erosion Filter Guided by Structural Generation Fields
    // Project 3D geocentric velocity into the local 2D plane
    vec2 local_vel = vec2(dot(vel, dP_dx), dot(vel, dP_dz));

    // Skew the slope vector so erosion tracks along the fold direction
    vec2 tectonic_skewed_slope = grad_local.xz + local_vel * 0.5;
    vec3 heightAndSlope = vec3(true_height, tectonic_skewed_slope);
    // vec3 heightAndSlope = vec3(true_height, grad_local.x, grad_local.z);

    // Derive erosion parameters dynamically from structural fields:
    // - High mountain ridges (fault_mask) & high elevation fade towards ridges (+1.0)
    // - Canyons & rivers carved channels fade towards creases (-1.0)
    // - Lake basins buffer towards flat sediment deposition (+0.3)
    float frameworkHeightPeak = config.max_height;
    float elevRatio = clamp(true_height / (frameworkHeightPeak * 0.6), -1.0, 1.0);
    float erosionFadeTarget = clamp(
        mix(elevRatio, 1.0, fault_mask * 0.5) - canyon_mask * 0.6 - river_profile * river_mask * 0.4 + basin_mask * 0.5,
        -1.0, 1.0
    );

    // Erosion strength increases in mountain/canyon fault regions and reduces in flat lake basins and water regions in general (?)
    float hydrology_protection = max(basin_mask * 0.7,
                                max(river_profile * river_mask,
                                    canyon_profile * canyon_mask));

    // float erosion_strength = mix(0.08, 0.20, clamp01(fault_mask * 0.7 + canyon_mask * 0.3)) * (1.0 - basin_mask * 0.7);
    float erosion_strength = mix(0.08, 0.20, clamp01(fault_mask * 0.7 + canyon_mask * 0.3)) * (1.0 - hydrology_protection);

    // Gully carving weight is sharper in steep mountain fault zones and canyon walls
    float erosion_gully_weight = mix(0.40, 0.70, clamp01(collision_intensity + canyon_mask * 0.5));
    // Detail restriction and spatial scale adjust according to ridge features
    float erosion_detail = mix(1.2, 1.8, clamp01(collision_intensity + canyon_mask));

    float erosion_scale = mix(0.12, 0.22, fault_mask);
    vec4 erosion_rounding = vec4(mix(1.0, 3.0, t_macro), mix(0.5, 0.0, t_macro), 0.1, 2.0);

    float ridgeMapVal = 0.0;
    float substrateVal = 0.0;
    float debugVal = 0.0;

    vec4 erosionDelta = ErosionFilter(
        p_local.xz,
        heightAndSlope,
        erosionFadeTarget,
        erosion_strength,
        erosion_gully_weight,
        erosion_detail,
        erosion_rounding,                                  // Rounding parameters (x: ridge, y: crease, z: input mult, w: octave mult)
        vec4(1.25, 1.25, 2.8, 1.5),                        // effect onset (x: initial, y: octave, z: ridgemap initial, w: ridgemap octave).
        normalize(vec2(dot(vel, dP_dx), dot(vel, dP_dz))), // assumedSlope
        erosion_scale,
        5,                                                 // octaves
        2.3,                                               // lacunarity
        0.52,                                              // gain
        1.70,                                              // cellScale
        0.5,                                               // normalization
        ridgeMapVal,
        substrateVal,
        debugVal
    );

    true_height += erosionDelta.x;
    vec2 newSlope = heightAndSlope.yz + erosionDelta.yz;
    out_normal = normalize(vec3(-newSlope.x, 1.0, poleFlipped ? newSlope.y : -newSlope.y));
    out_ridge_map = max(massive_ridge, ridgeMapVal);

    // Calculate water level in normalized height space
    float river_water_h = raw_final_h + river_carve * 0.85; // Rivers fill to 85% of carved depth
    float canyon_water_h = raw_final_h + canyon_carve * 0.05; // Canyons have shallow streams at the bottom

    float local_water_norm = max(
        lake_water_level * smoothstep(0.1, 0.5, basin_mask),
        max(
            river_water_h * smoothstep(0.01, 0.2, river_profile * river_mask),
            canyon_water_h * smoothstep(0.01, 0.1, canyon_profile * canyon_mask)
        )
    );

    float has_water_mask = max(smoothstep(0.1, 0.5, basin_mask),
                           max(smoothstep(0.01, 0.2, river_profile * river_mask),
                               smoothstep(0.01, 0.1, canyon_profile * canyon_mask)));

    // Plunge the water level 1000 units underground where there is no water presence
    out_water_level = mix(config.min_height - 1000.0, remap(local_water_norm, 0.0, 1.0, config.min_height, config.max_height), has_water_mask);

    return true_height;
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
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, out_water_level, dummyRidgeMap);
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
    return evaluate_terrain_analytical(p_local, phase, warp_strength, config, out_normal, dummyWaterLevel, dummyRidgeMap);
}