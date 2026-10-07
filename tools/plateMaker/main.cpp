
#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

#include "Simplex.h"
#include <glm/gtc/noise.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtx/matrix_operation.hpp>

// ==============================================================================
// CONFIGURATION & CONSTANTS
// ==============================================================================

struct TerrainConfig {
	float spatial_scale;
	float min_height;
	float max_height;
	float ridge_weight;
	float biome_bleed;
};

struct TerrainSample {
	float     height;
	glm::vec3 normal;
	float     ridgeMap;
	float     substrate;
};

struct TectonicPlate {
	glm::vec3 seed_dir;
	float     height;
	float     k;
	glm::vec3 velocity;
};

const float FAKE_PLANET_RADIUS = 600000.0f;
const float PHI = 1.618033988749894848204586834f;
const float TAU = 6.28318530717958647692f;

// Note: GLM matrix constructors are column-major, identical to GLSL
const glm::mat3 GOLD = glm::mat3(
	-0.571464913f,
	+0.814921382f,
	+0.096597072f,
	-0.278044873f,
	-0.303026659f,
	+0.911518454f,
	+0.772087367f,
	+0.494042493f,
	+0.399753815f
);

// ==============================================================================
// NOISE INTERFACES (Wire these to FastNoise2 or your existing generators)
// ==============================================================================

inline float remap(float v, float in_min, float in_max, float out_min, float out_max) {
	return out_min + (v - in_min) * (out_max - out_min) / (in_max - in_min);
}

inline float clamp01(float x) {
	return glm::clamp(x, 0.0f, 1.0f);
}

inline float ease_out(float x) {
	return 1.0f - (1.0f - x) * (1.0f - x);
}

inline float smooth_start(float x, float a) {
	return std::pow(std::max(0.0f, x), a);
}

inline float pow_inv(float x, float y) {
	return std::pow(std::max(0.0f, x), 1.0f / y);
}

inline glm::vec2 safe_normalize(glm::vec2 v) {
	return glm::length(v) > 0.0f ? glm::normalize(v) : glm::vec2(0.0f);
}

// Placeholder for missing noised / dot_noise_fbm
inline glm::vec4 noised(glm::vec3 p) {
	return glm::vec4(glm::simplex(p), 0.0f, 0.0f, 0.0f); // Replace with analytical derivative noise
}

inline float dot_noise_fbm(glm::vec3 p, int octaves, float phase, glm::vec3& out_grad) {
	out_grad = glm::vec3(0.0f); // Replace with your FBM implementation
	return glm::simplex(p);
}

glm::vec3 cross_noise(glm::vec3 p, float phase) {
	glm::vec3 rotated_p1 = GOLD * p;
	// GLSL `p * GOLD` translates cleanly in GLM as multiplication order is identical
	glm::vec3 rotated_p2 = PHI * (p * GOLD);

	glm::vec3 cos_phase = rotated_p1 + glm::vec3(phase, phase * 1.3f, phase * 1.7f);
	glm::vec3 sin_phase = rotated_p2 + glm::vec3(phase * 1.1f, phase * 0.7f, phase * 1.5f);

	return glm::cross(glm::cos(cos_phase), glm::sin(sin_phase));
}

glm::vec3 cross_noise_fbm(glm::vec3 p, float oct, float phase) {
	glm::vec3 val(0.0f);
	float     amp = 1.0f;
	float     freq = 1.0f;
	float     max_amp = 0.0f;
	for (int i = 0; i < std::max(0, static_cast<int>(oct)); i++) {
		val += amp * cross_noise(p * freq + val * freq, phase * freq);
		max_amp += amp;
		amp *= 0.5f;
		freq *= 2.0f;
	}
	return val / max_amp;
}

// Standard sine-based hash to bridge the GLSL `hash()` call.
// Replace this if your GLSL implementation uses a different PRNG (e.g., hash12 or hash22).
inline glm::vec2 hash(glm::vec2 p) {
	glm::vec2 q(glm::dot(p, glm::vec2(127.1f, 311.7f)), glm::dot(p, glm::vec2(269.5f, 183.3f)));
	return glm::fract(glm::sin(q) * 43758.5453123f);
}

// Should return a value roughly between -1.0 and 1.0
float sample_crustal_density(glm::vec3 p) {
	return Simplex::iqfBm(p);
}

// If FastNoise2 doesn't have native curl noise, you can approximate it
// by taking the cross product of the gradients of three distinct Simplex layers.
glm::vec3 sample_mantle_curl(glm::vec3 p) {
	return Simplex::curlNoise(p);
}

glm::vec4 PhacelleNoise(glm::vec2 p, glm::vec2 normDir, float freq, float offset, float normalization) {
	// The GLSL swizzle `normDir.yx * vec2(-1.0, 1.0)` resolves to `(-normDir.y, normDir.x)`
	glm::vec2 sideDir = glm::vec2(-normDir.y, normDir.x) * freq * TAU;
	offset *= TAU;

	glm::vec2 pInt = glm::floor(p);
	glm::vec2 pFrac = glm::fract(p);
	glm::vec2 phaseDir(0.0f);
	float     weightSum = 0.0f;

	for (int i = -1; i <= 2; i++) {
		for (int j = -1; j <= 2; j++) {
			glm::vec2 gridOffset(static_cast<float>(i), static_cast<float>(j));
			glm::vec2 gridPoint = pInt + gridOffset;
			glm::vec2 randomOffset = hash(gridPoint) * 0.5f;
			glm::vec2 vectorFromCellPoint = pFrac - gridOffset - randomOffset;

			// Bell-shaped weight function
			float sqrDist = glm::dot(vectorFromCellPoint, vectorFromCellPoint);
			float weight = std::exp(-sqrDist * 2.0f);
			weight = std::max(0.0f, weight - 0.01111f);

			weightSum += weight;
			float waveInput = glm::dot(vectorFromCellPoint, sideDir) + offset;

			// Add this cell's cosine and sine wave contributions
			phaseDir += glm::vec2(std::cos(waveInput), std::sin(waveInput)) * weight;
		}
	}

	glm::vec2 interpolated = phaseDir / weightSum;
	float     magnitude = std::sqrt(glm::dot(interpolated, interpolated));
	magnitude = std::max(1.0f - normalization, magnitude);

	// GLM allows constructing a vec4 from (vec2, vec2)
	return glm::vec4(interpolated / magnitude, sideDir);
}

// ==============================================================================
// CORE TECTONIC MATH
// ==============================================================================

// Returns the exact 'k' value needed to create a transition zone of 'width_km'
// Example uses:
// 10km sharp coastline transition -> k = ~10,700
// 50km wide mountain range      -> k = ~430
// 150km massive crumple zone    -> k = ~95
float calculate_k_for_width(float width_km, float planet_radius = 600.0f) {
	float theta = width_km / planet_radius;
	float denom = std::cos(theta) - 1.0f;

	// Prevent divide by zero if width is 0
	if (std::abs(denom) < 0.000001f)
		return 10000.0f;

	return std::log(0.05f) / denom;
}

std::vector<TectonicPlate> generate_tectonic_plates(int total_plates, int num_fracture_zones) {
	std::vector<TectonicPlate> plates;
	plates.reserve(total_plates);

	// 1. Distribute Seeds (Clustering Logic)
	int major_plates = total_plates / 2;
	int micro_plates = total_plates - major_plates;
	int micro_per_zone = micro_plates / std::max(1, num_fracture_zones);

	// Generate spread-out major plates
	for (int i = 0; i < major_plates; ++i) {
		TectonicPlate p;
		p.seed_dir = glm::sphericalRand(1.0f);
		plates.push_back(p);
	}

	// Generate clustered micro-plates in fracture zones
	for (int i = 0; i < num_fracture_zones; ++i) {
		glm::vec3 zone_center = glm::sphericalRand(1.0f);

		for (int j = 0; j < micro_per_zone; ++j) {
			// Generate a random vector within a small cone around the zone center
			glm::vec3 offset = glm::sphericalRand(0.2f); // 0.2 dictates cluster radius

			TectonicPlate p;
			p.seed_dir = glm::normalize(zone_center + offset);
			plates.push_back(p);
		}
	}

	// 2. Derive Physical Properties
	for (size_t i = 0; i < plates.size(); ++i) {
		glm::vec3 P = plates[i].seed_dir;

		// --- A. Base Height (Simplex) ---
		float crust_val = sample_crustal_density(P * 2.0f); // Low frequency

		if (crust_val < 0.0f) {
			// Oceanic Crust: Dense, sinks low (-1.0 to -0.2)
			plates[i].height = glm::mix(-0.2f, -1.0f, std::abs(crust_val));
		} else {
			// Continental Crust: Light, floats high (0.2 to 1.0)
			plates[i].height = glm::mix(0.2f, 1.0f, crust_val);
		}

		// --- B. Tectonic Velocity (Curl) ---
		// Sample the divergence-free mantle flow
		glm::vec3 global_flow = sample_mantle_curl(P * 1.5f);

		// Gram-Schmidt orthogonalization: project flow onto the sphere's tangent plane
		glm::vec3 radial_component = glm::dot(global_flow, P) * P;
		glm::vec3 tangent_flow = global_flow - radial_component;

		float drift_speed = 1.0f; // Scale to tune domain warping strength
		plates[i].velocity = glm::normalize(tangent_flow) * drift_speed;

		// --- C. Temperature / Sharpness (Nearest Neighbor) ---
		float min_dist = 999.0f;
		for (size_t j = 0; j < plates.size(); ++j) {
			if (i == j)
				continue;
			float dist = glm::distance(P, plates[j].seed_dir);
			min_dist = std::min(min_dist, dist);
		}

		// Map neighbor distance to k value
		// Closely packed plates get high k (sharp fjords/faults, e.g., 60.0)
		// Widely spread plates get low k (smooth craton borders, e.g., 15.0)
		float max_expected_dist = 1.0f; // Roughly 60 degrees apart
		float normalized_dist = glm::clamp(min_dist / max_expected_dist, 0.0f, 1.0f);

		plates[i].k = glm::mix(60.0f, 15.0f, normalized_dist);
	}

	return plates;
}

void evaluate_tectonics_geocentric(
	const std::vector<TectonicPlate>& plates,
	glm::vec3                         p_local,
	float                             k_crumple,
	float&                            out_base_h,
	glm::vec3&                        out_grad_h,
	float&                            out_fault,
	glm::vec3&                        out_grad_f,
	glm::vec3&                        out_vel,
	glm::mat3&                        out_J_vel,
	glm::vec3&                        P_geo,
	glm::vec3&                        dP_dx,
	glm::vec3&                        dP_dz
) {
	float theta = p_local.x / FAKE_PLANET_RADIUS;
	float phi = p_local.z / FAKE_PLANET_RADIUS;
	float sin_t = std::sin(theta);
	float cos_t = std::cos(theta);
	float sin_p = std::sin(phi);
	float cos_p = std::cos(phi);

	P_geo = glm::vec3(cos_p * sin_t, sin_p, cos_p * cos_t);
	dP_dx = glm::vec3(cos_p * cos_t, 0.0f, -cos_p * sin_t) / FAKE_PLANET_RADIUS;
	dP_dz = glm::vec3(-sin_p * sin_t, cos_p, -sin_p * cos_t) / FAKE_PLANET_RADIUS;

	// Pass 1: Dual Log-Sum-Exp Trick
	float max_kd_macro = -1e20f;
	float max_kd_crump = -1e20f;
	for (const auto& plate : plates) {
		float d = glm::dot(P_geo, plate.seed_dir);
		max_kd_macro = std::max(max_kd_macro, plate.k * d);
		max_kd_crump = std::max(max_kd_crump, k_crumple * d);
	}

	float     sum_w_macro = 0.0f;
	float     sum_h = 0.0f;
	glm::vec3 grad_w_macro(0.0f);
	glm::vec3 grad_h(0.0f);

	float     sum_w_crump = 0.0f;
	glm::vec3 sum_v(0.0f);
	float     sum_w2_crump = 0.0f;
	glm::vec3 grad_w_crump(0.0f);
	glm::mat3 J_sum_v(0.0f);
	glm::vec3 grad_w2_crump(0.0f);

	// Pass 2: Simultaneous Accumulation
	for (const auto& plate : plates) {
		glm::vec3 S = plate.seed_dir;
		float     d = glm::dot(P_geo, S);

		// Sharp Macro Evaluation
		float     w_macro = std::exp(plate.k * d - max_kd_macro);
		glm::vec3 dw_macro = plate.k * w_macro * S;

		sum_w_macro += w_macro;
		sum_h += w_macro * plate.height;
		grad_w_macro += dw_macro;
		grad_h += plate.height * dw_macro;

		// Wide Crumple Zone Evaluation
		float     w_crump = std::exp(k_crumple * d - max_kd_crump);
		glm::vec3 dw_crump = k_crumple * w_crump * S;

		sum_w_crump += w_crump;
		sum_v += w_crump * plate.velocity;
		grad_w_crump += dw_crump;
		J_sum_v += glm::outerProduct(plate.velocity, dw_crump);

		sum_w2_crump += w_crump * w_crump;
		grad_w2_crump += 2.0f * w_crump * dw_crump;
	}

	// Resolve Sharp Height
	float inv_W_macro = 1.0f / sum_w_macro;
	out_base_h = sum_h * inv_W_macro;
	out_grad_h = (grad_h - out_base_h * grad_w_macro) * inv_W_macro;

	// Resolve Wide Velocity Field
	float inv_W_crump = 1.0f / sum_w_crump;
	float inv_W_crump2 = inv_W_crump * inv_W_crump;

	out_vel = sum_v * inv_W_crump;
	out_J_vel = (J_sum_v - glm::outerProduct(out_vel, grad_w_crump)) * inv_W_crump;

	// Fault Mask Evaluation
	float sum_N2_crump = sum_w2_crump * inv_W_crump2;
	float raw_fault = 1.0f - sum_N2_crump;
	float max_fault_expected = 0.55f;

	float t_fault = glm::clamp(raw_fault / max_fault_expected, 0.0f, 1.0f);
	out_fault = t_fault * t_fault * (3.0f - 2.0f * t_fault);

	float     d_smooth_fault = 6.0f * t_fault * (1.0f - t_fault) / max_fault_expected;
	glm::vec3 raw_grad_f = -(grad_w2_crump - 2.0f * sum_N2_crump * sum_w_crump * grad_w_crump) * inv_W_crump2;
	out_grad_f = raw_grad_f * d_smooth_fault;
}

float evaluate_terrain_analytical(
	const std::vector<TectonicPlate>& plates,
	glm::vec3                         p_local,
	float                             phase,
	float                             warp_strength,
	const TerrainConfig&              config,
	glm::vec3&                        out_normal
) {
	glm::vec3 p_tectonic = p_local + 20000.0f * cross_noise_fbm(4.0f * p_local / FAKE_PLANET_RADIUS, 6.0f, 0.0f);

	float     base_h, fault_mask;
	glm::vec3 grad_base_h, grad_fault, vel, P_geo, dP_dx, dP_dz;
	glm::mat3 J_vel;

	float k_crumple = 450.0f;

	evaluate_tectonics_geocentric(
		plates,
		p_tectonic,
		k_crumple,
		base_h,
		grad_base_h,
		fault_mask,
		grad_fault,
		vel,
		J_vel,
		P_geo,
		dP_dx,
		dP_dz
	);

	glm::mat3 S = 0.5f * (J_vel + glm::transpose(J_vel));
	float     I1 = S[0][0] + S[1][1] + S[2][2];

	float collision_intensity = glm::smoothstep(-0.5f, -2.0f, I1);
	float uplift = collision_intensity * 0.4f;
	base_h += uplift;

	float     radius_scale = FAKE_PLANET_RADIUS * config.spatial_scale;
	glm::vec3 P_noise = P_geo * radius_scale;

	glm::vec4 raw_base_noise = noised(P_noise);
	float     base_noise = raw_base_noise.x;
	glm::vec3 grad_base_noise = glm::vec3(raw_base_noise.y, raw_base_noise.z, raw_base_noise.w);

	float     base_noise_amp = 0.95f;
	float     continent_h = base_noise * base_noise_amp;
	glm::vec3 grad_continent_h = grad_base_noise * base_noise_amp;

	glm::vec3 P_warped = P_noise + vel * warp_strength;
	glm::mat3 J_warp = glm::mat3(radius_scale) + J_vel * warp_strength;

	glm::vec3 grad_raw_ridge(0.0f);
	float     raw_ridge = dot_noise_fbm(P_warped, 6, phase + 42.0f, grad_raw_ridge);
	glm::vec3 grad_ridge_warped = glm::transpose(J_warp) * grad_raw_ridge;

	float     ridge_h = raw_ridge * fault_mask;
	glm::vec3 grad_ridge = (grad_ridge_warped * fault_mask) + (raw_ridge * grad_fault);

	float     final_h = base_h + continent_h + ridge_h;
	glm::vec3 final_grad_geo = grad_base_h + grad_continent_h + grad_ridge;

	float true_height = remap(final_h, 0.0f, 1.0f, config.min_height, config.max_height);
	float height_amplitude = config.max_height - config.min_height;

	final_grad_geo *= height_amplitude;

	glm::vec3 grad_local = glm::vec3(glm::dot(final_grad_geo, dP_dx), 0.0f, glm::dot(final_grad_geo, dP_dz));

	out_normal = glm::normalize(glm::vec3(-grad_local.x, 1.0f, -grad_local.z));
	return true_height;
}

glm::vec4 ErosionFilter(
	glm::vec2 p,
	glm::vec3 heightAndSlope,
	float     fadeTarget,
	float     strength,
	float     gullyWeight,
	float     detail,
	glm::vec4 rounding,
	glm::vec4 onset,
	glm::vec2 assumedSlope,
	float     scale,
	int       octaves,
	float     lacunarity,
	float     gain,
	float     cellScale,
	float     normalization,
	float&    ridgeMap,
	float&    substrate,
	float&    debug
) {
	strength *= (scale + 10.0f);
	fadeTarget = glm::clamp(fadeTarget, -1.0f, 1.0f);
	scale *= 100.0f;

	glm::vec3 inputHeightAndSlope = heightAndSlope;
	float     freq = 1.0f / (scale * cellScale);
	float     slopeLength = std::max(glm::length(glm::vec2(heightAndSlope.y, heightAndSlope.z)), 1e-10f);
	float     magnitude = 0.0f;
	float     roundingMult = 1.0f;

	float roundingForInput = glm::mix(rounding.y, rounding.x, clamp01(fadeTarget + 0.5f)) * rounding.z;
	float combiMask = ease_out(smooth_start(slopeLength * onset.x, roundingForInput * onset.x));

	float ridgeMapCombiMask = ease_out(slopeLength * onset.z);
	float ridgeMapFadeTarget = fadeTarget;

	glm::vec2 gullySlope = glm::mix(
		glm::vec2(heightAndSlope.y, heightAndSlope.z),
		glm::vec2(heightAndSlope.y, heightAndSlope.z) / slopeLength * assumedSlope.x,
		assumedSlope.y
	);

	for (int i = 0; i < octaves; i++) {
		glm::vec4 phacelle = PhacelleNoise(p * freq, safe_normalize(gullySlope), cellScale, 0.25f, normalization);
		phacelle.z *= -freq;
		phacelle.w *= -freq;
		float sloping = std::abs(phacelle.y);

		gullySlope += glm::sign(phacelle.y) * glm::vec2(phacelle.z, phacelle.w) * strength * gullyWeight;

		glm::vec3 gullies = glm::vec3(phacelle.x, phacelle.y * phacelle.z, phacelle.y * phacelle.w);
		glm::vec3 fadedGullies = glm::mix(glm::vec3(fadeTarget, 0.0f, 0.0f), gullies * gullyWeight, combiMask);
		heightAndSlope += fadedGullies * strength;
		magnitude += strength;

		fadeTarget = fadedGullies.x;

		float roundingForOctave = glm::mix(rounding.y, rounding.x, clamp01(phacelle.x + 0.5f)) * roundingMult;
		float newMask = ease_out(smooth_start(sloping * onset.y, roundingForOctave * onset.y));
		combiMask = pow_inv(combiMask, detail) * newMask;

		ridgeMapFadeTarget = glm::mix(ridgeMapFadeTarget, gullies.x, ridgeMapCombiMask);
		float newRidgeMapMask = ease_out(sloping * onset.w);
		ridgeMapCombiMask = ridgeMapCombiMask * newRidgeMapMask;

		strength *= gain;
		freq *= lacunarity;
		roundingMult *= rounding.w;
	}

	ridgeMap = ridgeMapFadeTarget * (1.0f - ridgeMapCombiMask);
	substrate = glm::clamp(fadeTarget, -1.0f, 1.0f);
	debug = fadeTarget;

	return glm::vec4(heightAndSlope - inputHeightAndSlope, magnitude);
}

// ==============================================================================
// C++ ENTRY POINT
// ==============================================================================

TerrainSample evaluate_point(glm::vec2 worldXZ, const std::vector<TectonicPlate>& plates) {
	TerrainConfig con = {0.00001f, -100.0f, 1000.0f, 0.14f, 0.5f};
	glm::vec3     p = glm::vec3(worldXZ.x, 0.0f, worldXZ.y);

	glm::vec3 normal(0.0f, 1.0f, 0.0f);
	float     h = evaluate_terrain_analytical(plates, p, 0.1f, 0.75f, con, normal);

	float     ny = std::max(0.0001f, normal.y);
	glm::vec3 heightAndSlope = glm::vec3(h, -normal.x / ny, -normal.z / ny);

	float ridgeMap = 0.0f, substrate = 0.0f, debugVal = 0.0f;

	glm::vec4 erosionDelta = ErosionFilter(
		worldXZ,
		heightAndSlope,
		0.0f,
		0.05f,
		0.5f,
		1.0f,
		glm::vec4(0.5f, 0.5f, 1.0f, 1.0f),
		glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
		glm::vec2(1.0f, 0.0f),
		0.01f,
		4,
		2.0f,
		0.5f,
		1.0f,
		1.0f,
		ridgeMap,
		substrate,
		debugVal
	);

	h += erosionDelta.x;
	glm::vec2 newSlope = glm::vec2(heightAndSlope.y, heightAndSlope.z) + glm::vec2(erosionDelta.y, erosionDelta.z);

	TerrainSample sample;
	sample.height = h;
	sample.normal = glm::normalize(glm::vec3(-newSlope.x, 1.0f, -newSlope.y));
	sample.ridgeMap = ridgeMap;
	sample.substrate = substrate;
	return sample;
}

int main() {
	auto plates = generate_tectonic_plates(24, 2);
	for (auto i : plates) {
		fmt::println(
			"TectonicPlate(vec3({}, {}, {}), {}, {}, vec3({}, {}, {})),",
			i.seed_dir.x,
			i.seed_dir.y,
			i.seed_dir.z,
			i.height,
			i.k,
			i.velocity.x,
			i.velocity.y,
			i.velocity.z
		);
	}
}