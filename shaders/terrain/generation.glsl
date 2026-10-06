#include "common.glsl"
#include "terrain.glsl"
#include "lygia/generative/psrdnoise.glsl"
#include "lygia/generative/noised.glsl"


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

// Derivative of smoothstep(edge0, edge1, x)
float d_smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return 6.0 * t * (1.0 - t) / (edge1 - edge0);
}

float evaluate_terrain_analytical(vec3 p, float phase, float warp_strength, TerrainConfig config, out vec3 out_normal, out float out_biome, out float out_mask) {
    p *= config.spatial_scale;

    // 1. Continent Mask (Macro Structure)
    // Replaces divergence (I1). We use the base value of a low-frequency scalar noise.
    vec3 g_mask;
    float v_mask = dot_noise_fbm(p, 2, phase, g_mask);

    float continent_mask = 1.0 - smoothstep(-0.5, 0.2, v_mask);

    // Chain rule: d(mask)/dp = d(smoothstep)/dv * d(noise)/dp
    float d_mask_dv = -d_smoothstep(-0.5, 0.2, v_mask);
    vec3 grad_continent_mask = d_mask_dv * g_mask;

    // 2. Domain Warping (Deformation)
    // Replaces the strain tensor. We warp using the gradient of a noise field.
    vec3 g_warp;
    mat3 h_warp;
    // Evaluate noise and extract both its gradient and its Hessian (2nd derivative)
    float v_warp = dot_noise_fbm(p, 2, phase + 13.37, g_warp, h_warp);

    vec3 p_warped = p + g_warp * warp_strength;

    // Jacobian matrix of the domain warp: J = I + Hessian * strength
    mat3 J_warp = mat3(1.0) + h_warp * warp_strength;

    // 3. Base Terrain Elevation
    vec3 g_base;
    float v_base = dot_noise_fbm(p, 4, phase, g_base);

    // 4. Ridge Terrain (Evaluated in warped space)
    vec3 g_raw_ridge;
    float v_raw_ridge = dot_noise_fbm(p_warped, 6, phase + 42.0, g_raw_ridge);

    // Chain rule for warped domain: grad(Ridge(p_warped)) = J_warp^T * grad(Ridge)
    vec3 g_ridge_warped = transpose(J_warp) * g_raw_ridge;

    // Ridged multifractal mapping: height = 1.0 - abs(v)
    // The mathematical derivative of abs(x) is sign(x).
    // *Note*: If you are using your `filtered_abs` anti-aliasing logic here,
    // replace `sign()` with the analytical derivative of your smoothed absolute function.
    float ridge_height = 1.0 - abs(v_raw_ridge);
    vec3 g_ridge_final = -sign(v_raw_ridge) * g_ridge_warped;

    // 5. Composition (Product Rule)
    // Combined Height = (Base + Ridge * 0.5) * Mask
    float combined_height = v_base + (ridge_height * 0.5);
    vec3 grad_combined = g_base + (g_ridge_final * 0.5);

    // Apply the product rule: d(u*v) = u'*v + u*v'
    float final_height = combined_height * continent_mask;
    vec3 grad_final = grad_combined * continent_mask + combined_height * grad_continent_mask;

    // 6. Biomes
    // Use the gradient magnitude of the warp field as a stand-in for shear stress (I2)
    float base_rocky = smoothstep(0.0, 0.8, length(g_warp));
    vec3 dominant_axis = normalize(g_warp + vec3(0.0001)); // Prevent division by zero

    vec3 biome_coords = vec3(1.0 - final_height, v_mask * 0.5 + 0.5, base_rocky);
    float biome_dither = v_raw_ridge * config.biome_bleed;
    biome_coords += dominant_axis * biome_dither;

    out_biome = biome_map(clamp(biome_coords.x, 0.0, 1.0), clamp(biome_coords.y, 0.0, 1.0), clamp(biome_coords.z, 0.0, 1.0));
    out_mask = continent_mask;

    // 7. Resolve Final Elevation & Normal
    float true_height = remap(final_height, 0.0, 1.0, config.min_height, config.max_height);

    // Scale the gradient by the spatial scale and the remap height amplitude
    float height_amplitude = config.max_height - config.min_height;
    vec3 scaled_grad = grad_final * height_amplitude * config.spatial_scale;

    // Assuming a planar/tangent-space projection mapping where Y is up
    out_normal = normalize(vec3(-scaled_grad.x, 1.0, -scaled_grad.z));

    return true_height;
}

struct TectonicPlate {
    vec3 seed_dir; // Normalized direction of the plate center
    float height;  // Base continent elevation (e.g., 1.0 for land, 0.0 for ocean)
	float k;
    vec3 velocity;
};

// // SSBO containing your continent seeds
// layout(std430, binding = 0) readonly buffer PlateBuffer {
//     TectonicPlate plates[];
// };

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

// k = Sharpness of the plate boundaries.
// Higher k = sharper tectonic faults. Lower k = smoother transitions.
void evaluate_soft_voronoi(vec3 p, float k, int num_plates, out float out_height, out vec3 out_grad) {
    vec3 P_norm = normalize(p);

    // Pass 1: Find the maximum dot product to prevent exp() overflow (Log-Sum-Exp trick)
    float max_dot = -1.0;
    for(int i = 0; i < num_plates; i++) {
        float d = dot(P_norm, plates[i].seed_dir);
        max_dot = max(max_dot, d);
    }

    // Pass 2: Accumulate the Softmax values and exact derivatives
    float sum_weight = 0.0;
    float sum_height = 0.0;

    vec3 grad_weight = vec3(0.0);
    vec3 grad_height = vec3(0.0);

    for(int i = 0; i < num_plates; i++) {
        vec3 S = plates[i].seed_dir;
        float V = plates[i].height;

        // The proximity metric
        float d = dot(P_norm, S);

        // Stabilized exponential weight
        float w = exp(k * (d - max_dot));

        // Accumulate denominators (weights) and numerators (weighted values)
        sum_weight += w;
        sum_height += w * V;

        // Accumulate derivatives
        // Derivative of exp(k * d) with respect to P is k * S * exp(k * d)
        vec3 dw = k * w * S;

        grad_weight += dw;
        grad_height += V * dw;
    }

    // Final Value: Weighted average
    out_height = sum_height / sum_weight;

    // Final Gradient: Quotient Rule -> d(N/D) = (D*dN - N*dD) / D^2
    // Which algebraically simplifies to -> (dN - Height * dD) / D
    out_grad = (grad_height - out_height * grad_weight) / sum_weight;
}


// Ensure FAKE_PLANET_RADIUS is in scope
void evaluate_soft_voronoi_pseudosphere(vec3 p, float k, int num_plates, out float out_height, out vec3 out_grad_local) {
    // 1. Map local XZ distances to radians (Longitude/Latitude)
    float theta = p.x / FAKE_PLANET_RADIUS; // Longitude
    float phi   = p.z / FAKE_PLANET_RADIUS; // Latitude

    float sin_theta = sin(theta); float cos_theta = cos(theta);
    float sin_phi   = sin(phi);   float cos_phi   = cos(phi);

    // 2. Construct the geocentric unit vector
    vec3 P_geo = vec3(
        cos_phi * sin_theta,
        sin_phi,
        cos_phi * cos_theta
    );

    // 3. Evaluate the Softmax Voronoi (identical log-sum-exp logic)
    float max_dot = -1.0;
    for(int i = 0; i < num_plates; i++) {
        max_dot = max(max_dot, dot(P_geo, plates[i].seed_dir));
    }

    float sum_weight = 0.0;
    float sum_height = 0.0;
    vec3 grad_weight = vec3(0.0);
    vec3 grad_height = vec3(0.0);

    for(int i = 0; i < num_plates; i++) {
        vec3 S = plates[i].seed_dir;
        float V = plates[i].height;

        float d = dot(P_geo, S);
        float w = exp(k * (d - max_dot));

        sum_weight += w;
        sum_height += w * V;

        vec3 dw = k * w * S;
        grad_weight += dw;
        grad_height += V * dw;
    }

    out_height = sum_height / sum_weight;
    vec3 grad_geo = (grad_height - out_height * grad_weight) / sum_weight;

    // 4. Translate the gradient back to the flat pseudo-sphere domain
    // Partial derivative of P_geo with respect to surface distance X
    vec3 dP_dx = vec3(
         cos_phi * cos_theta,
         0.0,
        -cos_phi * sin_theta
    ) / FAKE_PLANET_RADIUS;

    // Partial derivative of P_geo with respect to surface distance Z
    vec3 dP_dz = vec3(
        -sin_phi * sin_theta,
         cos_phi,
        -sin_phi * cos_theta
    ) / FAKE_PLANET_RADIUS;

    // Dot the geocentric gradient with the Jacobian basis vectors
    out_grad_local = vec3(
        dot(grad_geo, dP_dx),
        0.0, // Elevation (Y) does not influence the macro continent layout
        dot(grad_geo, dP_dz)
    );
}

void evaluate_soft_voronoi_pseudosphere2(vec3 p, int num_plates, out float out_height, out vec3 out_grad_local) {
    float theta = p.x / FAKE_PLANET_RADIUS;
    float phi   = p.z / FAKE_PLANET_RADIUS;

    float sin_theta = sin(theta); float cos_theta = cos(theta);
    float sin_phi   = sin(phi);   float cos_phi   = cos(phi);

    vec3 P_geo = vec3(cos_phi * sin_theta, sin_phi, cos_phi * cos_theta);

    // Pass 1: Log-Sum-Exp Trick adapted for per-plate 'k'
    float max_kd = -1e20; // Must be very low, as (k * dot) can be highly negative
    for(int i = 0; i < num_plates; i++) {
        float kd = plates[i].k * dot(P_geo, plates[i].seed_dir);
        max_kd = max(max_kd, kd);
    }

    float sum_weight = 0.0;
    float sum_height = 0.0;
    vec3 grad_weight = vec3(0.0);
    vec3 grad_height = vec3(0.0);

    // Pass 2: Accumulation
    for(int i = 0; i < num_plates; i++) {
        vec3 S = plates[i].seed_dir;
        float V = plates[i].height;
        float k_i = plates[i].k;

        // Calculate the exponent with the specific plate's 'k'
        float kd = k_i * dot(P_geo, S);
        float w = exp(kd - max_kd);

        sum_weight += w;
        sum_height += w * V;

        // The chain rule pulls k_i out of the exponent
        vec3 dw = k_i * w * S;

        grad_weight += dw;
        grad_height += V * dw;
    }

    out_height = sum_height / sum_weight;
    vec3 grad_geo = (grad_height - out_height * grad_weight) / sum_weight;

    // Pass 3: Project back to pseudo-sphere Jacobian
    vec3 dP_dx = vec3( cos_phi * cos_theta, 0.0, -cos_phi * sin_theta) / FAKE_PLANET_RADIUS;
    vec3 dP_dz = vec3(-sin_phi * sin_theta, cos_phi, -sin_phi * cos_theta) / FAKE_PLANET_RADIUS;

    out_grad_local = vec3(dot(grad_geo, dP_dx), 0.0, dot(grad_geo, dP_dz));
}


void eval_terrain(vec3 p, TerrainConfig config, out float out_height, out vec3 out_normal) {
	float continent_mask;
	vec3 grad_continent_mask;
    vec3 p_scale = vec3(1.0 / (2.0 * PI * FAKE_PLANET_RADIUS), 1.0, 1.0 / (PI * FAKE_PLANET_RADIUS));
	evaluate_soft_voronoi_pseudosphere2(p, 32, continent_mask, grad_continent_mask);
	// p *= config.spatial_scale;
	p *= p_scale * 30.0;

	vec3 grad_noise;
	float warp = psrdnoise(p*continent_mask, vec3(0), 0.0);

	float noise_height = dot_noise_fbm(p+warp, 4, 0.0, grad_noise);

	vec3 grad_noise2;
	float warp2 = psrdnoise(p/continent_mask, vec3(0), 10.0);
	float noise_height2 = dot_noise_fbm(p*warp2, 4, 3.0, grad_noise2);

	grad_noise += grad_noise2;
	noise_height += noise_height2;

	float final_height = noise_height * continent_mask;
	vec3 final_grad = (grad_noise * continent_mask) + (noise_height * grad_continent_mask);

	float height_amplitude = config.max_height - config.min_height;
	vec3 scaled_grad = final_grad * height_amplitude * config.spatial_scale;

	out_normal = normalize(vec3(-scaled_grad.x, 1.0, -scaled_grad.z));
	out_height = remap(final_height, 0.0, 1.0, config.min_height, config.max_height);
}


struct TectonicPlate2 {
    vec3 seed_dir; // Normalized direction of the plate center
    float height;  // Base continent elevation (e.g., 1.0 for land, 0.0 for ocean)
	float k;
    vec3 velocity; // 3D Geocentric drift direction
    // You could also store vec3 drift_velocity here for tectonic flow
};

    // evaluate_tectonics_geocentric(p_tectonic, 32, k_crumple,
    //     base_h, grad_base_h,
    //     fault_mask, grad_fault,
    //     vel, J_vel,
    //     P_geo, dP_dx, dP_dz);


void evaluate_tectonics_geocentric(
    vec3 p_local, float k_crumple,
    out float out_base_h, out vec3 out_grad_h,
    out float out_fault,  out vec3 out_grad_f, // Added missing outputs
    out vec3 out_vel,     out mat3 out_J_vel,
    out vec3 P_geo, out vec3 dP_dx, out vec3 dP_dz)
{
    float theta = p_local.x / FAKE_PLANET_RADIUS;
    float phi   = p_local.z / FAKE_PLANET_RADIUS;
    float sin_t = sin(theta); float cos_t = cos(theta);
    float sin_p = sin(phi);   float cos_p = cos(phi);

    P_geo = vec3(cos_p * sin_t, sin_p, cos_p * cos_t);
    dP_dx = vec3( cos_p * cos_t, 0.0, -cos_p * sin_t) / FAKE_PLANET_RADIUS;
    dP_dz = vec3(-sin_p * sin_t, cos_p, -sin_p * cos_t) / FAKE_PLANET_RADIUS;

    // Pass 1: Dual Log-Sum-Exp Trick
    float max_kd_macro = -1e20;
    float max_kd_crump = -1e20;
    for(int i = 0; i < plates.length(); i++) {
        float d = dot(P_geo, plates[i].seed_dir);
        max_kd_macro = max(max_kd_macro, plates[i].k * d);
        max_kd_crump = max(max_kd_crump, k_crumple * d);
    }

    // Accumulators for Sharp Macro (Base Height)
    float sum_w_macro = 0.0; float sum_h = 0.0;
    vec3 grad_w_macro = vec3(0.0); vec3 grad_h = vec3(0.0);

    // Accumulators for Wide Crumple Zone (Velocity, Deformation & Faults)
    float sum_w_crump = 0.0;  vec3 sum_v = vec3(0.0);
    float sum_w2_crump = 0.0; // Squared weight accumulator
    vec3 grad_w_crump = vec3(0.0); mat3 J_sum_v = mat3(0.0);
    vec3 grad_w2_crump = vec3(0.0); // Squared weight gradient accumulator

    // Pass 2: Simultaneous Accumulation
    for(int i = 0; i < plates.length(); i++) {
        vec3 S = plates[i].seed_dir;
        float d = dot(P_geo, S);

        // A. Sharp Macro Evaluation
        float w_macro = exp(plates[i].k * d - max_kd_macro);
        vec3 dw_macro = plates[i].k * w_macro * S;

        sum_w_macro  += w_macro;
        sum_h        += w_macro * plates[i].height;
        grad_w_macro += dw_macro;
        grad_h       += plates[i].height * dw_macro;

        // B. Wide Crumple Zone Evaluation
        float w_crump = exp(k_crumple * d - max_kd_crump);
        vec3 dw_crump = k_crumple * w_crump * S;

        vec3 randVec = cross_noise(plates[i].seed_dir, plates[i].k);
        // vec3 V = 50*(randVec - plates[i].seed_dir * dot(randVec, plates[i].seed_dir));

        sum_w_crump  += w_crump;
        sum_v        += w_crump * plates[i].velocity;
        // sum_v        += w_crump * V;
        grad_w_crump += dw_crump;
        J_sum_v      += outerProduct(plates[i].velocity, dw_crump);
        // J_sum_v      += outerProduct(V, dw_crump);

        // Track squared weights and their gradients strictly for the Fault Line Mask
        sum_w2_crump += w_crump * w_crump;
        grad_w2_crump += 2.0 * w_crump * dw_crump;
    }

    // Resolve Sharp Height
    float inv_W_macro = 1.0 / sum_w_macro;
    out_base_h = sum_h * inv_W_macro;
    out_grad_h = (grad_h - out_base_h * grad_w_macro) * inv_W_macro;

    // Resolve Wide Velocity Field & Fault Mask
    float inv_W_crump = 1.0 / sum_w_crump;
    float inv_W_crump2 = inv_W_crump * inv_W_crump;

    out_vel   = sum_v * inv_W_crump;
    out_J_vel = (J_sum_v - outerProduct(out_vel, grad_w_crump)) * inv_W_crump;

    // Fault Mask: 1.0 - sum( (w/W)^2 )
    float sum_N2_crump = sum_w2_crump * inv_W_crump2;
// Old: out_fault = 1.0 - sum_N2_crump;

// New: Remap the raw mask so that 0.5 becomes 1.0 using a smooth curve
float raw_fault = 1.0 - sum_N2_crump;
float max_fault_expected = 0.55; // Slightly above 0.5 to account for 3-plate intersections

// smoothstep(0.0, max_fault_expected, raw_fault)
float t_fault = clamp(raw_fault / max_fault_expected, 0.0, 1.0);
out_fault = t_fault * t_fault * (3.0 - 2.0 * t_fault);

// Chain rule for the new smoothstep mask
float d_smooth_fault = 6.0 * t_fault * (1.0 - t_fault) / max_fault_expected;

vec3 raw_grad_f = -(grad_w2_crump - 2.0 * sum_N2_crump * sum_w_crump * grad_w_crump) * inv_W_crump2;
out_grad_f = raw_grad_f * d_smooth_fault;
}

// float radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
//     vec3 warp_v = vec3(0.0);
// 	float warp_s = psrdnoise(2*p_local/FAKE_PLANET_RADIUS, vec3(0), 0.0, warp_v);

float evaluate_terrain_analytical(vec3 p_local, float phase, float warp_strength, TerrainConfig config, out vec3 out_normal) {

    // -- PRE-WARP --
    // Jittering the input grid breaks up the mathematical linearity of the Voronoi cells
    // vec3 warp_v = vec3(0.0);
    // float warp_s = psrdnoise(4*p_local / FAKE_PLANET_RADIUS, vec3(0.0), 0.0, warp_v);
    // vec3 p_tectonic = p_local + 20000.0 * smoothstep(-0.5, 0.75, warp_s) * normalize(vec3(warp_v.z, 0.0, warp_v.x));

	vec3 p_tectonic = p_local + 20000.0 * cross_noise_fbm(4*p_local / FAKE_PLANET_RADIUS, 6, 0.0);

    // 1. Evaluate Dual-Temperature Tectonics
    float base_h, fault_mask;
    vec3 grad_base_h, grad_fault, vel, P_geo, dP_dx, dP_dz;
    mat3 J_vel;

    // A low k_crumple (e.g., 8.0 - 15.0) spreads the fault_mask and velocity field
    // hundreds of kilometers wide, while coastlines remain sharp.
    float k_crumple = 450.0;

    evaluate_tectonics_geocentric(p_tectonic, k_crumple,
        base_h, grad_base_h,
        fault_mask, grad_fault,
        vel, J_vel,
        P_geo, dP_dx, dP_dz);

    // 2. Extract Strain & Divergence from the Smooth Velocity Field
    mat3 S = 0.5 * (J_vel + transpose(J_vel));

    // I1 (Divergence): Negative = Collision (Mountains), Positive = Rift (Trenches)
    float I1 = S[0][0] + S[1][1] + S[2][2];

    // Continental Uplift: Gently raise the macro terrain where plates collide.
    // Because k_crumple is such a low-frequency, ultra-smooth field, we can safely
    // treat the gradient of this specific uplift layer as near-zero to avoid
    // needing the Hessian of the Voronoi cells, without causing visible lighting errors.
// Isolate violent collisions (highly negative I1)
// Adjust the -2.0 based on the magnitude of your velocity drift speed
float collision_intensity = smoothstep(-0.5, -2.0, I1);

// Aggressively push the base crust up.
// A multiplier of 0.4 means a collision can push a lowland plate (0.2)
// all the way up to a highland plateau (0.6) before any noise is added.
float uplift = collision_intensity * 0.4;
base_h += uplift;

    // 3. Base Continent Topography (Rolling Hills & Plains)
    float radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
    vec3 P_noise = P_geo * radius_scale;

    vec3 grad_base_noise;
    // float base_noise = dot_noise_fbm(P_noise, 4, phase, grad_base_noise);
	vec4 raw_base_noise = noised(P_noise);
	float base_noise = raw_base_noise.x;
	grad_base_noise = raw_base_noise.yzw;

    float base_noise_amp = 0.95;
    float continent_h = base_noise * base_noise_amp;
    vec3 grad_continent_h = grad_base_noise * base_noise_amp;

    // 4. Tectonic Domain Warping & Ridges (Wide Crumple Zones)
    vec3 P_warped = P_noise + vel * warp_strength;
    mat3 J_warp = mat3(radius_scale) + J_vel * warp_strength;

    vec3 grad_raw_ridge;
    float raw_ridge = dot_noise_fbm(P_warped, 6, phase + 42.0, grad_raw_ridge);
    vec3 grad_ridge_warped = transpose(J_warp) * grad_raw_ridge;

    // Product rule for the WIDE fault-masked ridges
    // Because fault_mask now comes from k_crumple, the mountain ranges will naturally
    // spill out far beyond the sharp tectonic borders.
    float ridge_h = raw_ridge * fault_mask;
    vec3 grad_ridge = (grad_ridge_warped * fault_mask) + (raw_ridge * grad_fault);

    // 5. Final Composition
    float final_h = base_h + continent_h + ridge_h;
    vec3 final_grad_geo = grad_base_h + grad_continent_h + grad_ridge;

    // 6. Resolve to Local Normal
    float true_height = remap(final_h, 0.0, 1.0, config.min_height, config.max_height);
    float height_amplitude = config.max_height - config.min_height;

    final_grad_geo *= height_amplitude;

    // Project the 3D Geocentric gradient back to the 2D pseudo-sphere tangents
    vec3 grad_local = vec3(
        dot(final_grad_geo, dP_dx),
        0.0,
        dot(final_grad_geo, dP_dz)
    );

    out_normal = normalize(vec3(-grad_local.x, 1.0, -grad_local.z));

    return true_height;
}
