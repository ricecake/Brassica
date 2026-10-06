#include <algorithm>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>

#include "Simplex.h"
#include <glm/gtc/random.hpp>
#include <fmt/core.h>

struct TectonicPlate {
	glm::vec3 seed_dir;
	float     height;
	float     k;
	glm::vec3 velocity;
};

// ==============================================================================
// NOISE INTERFACES (Wire these to FastNoise2 or your existing generators)
// ==============================================================================

// Should return a value roughly between -1.0 and 1.0
float sample_crustal_density(glm::vec3 p) {
	return Simplex::iqfBm(p);
}

// If FastNoise2 doesn't have native curl noise, you can approximate it
// by taking the cross product of the gradients of three distinct Simplex layers.
glm::vec3 sample_mantle_curl(glm::vec3 p) {
	return Simplex::curlNoise(p);
}

// Returns the exact 'k' value needed to create a transition zone of 'width_km'
// Example uses:
// 10km sharp coastline transition -> k = ~10,700
// 50km wide mountain range      -> k = ~430
// 150km massive crumple zone    -> k = ~95
float calculate_k_for_width(float width_km, float planet_radius = 600.0f) {
    float theta = width_km / planet_radius;
    float denom = std::cos(theta) - 1.0f;

    // Prevent divide by zero if width is 0
    if (std::abs(denom) < 0.000001f) return 10000.0f;

    return std::log(0.05f) / denom;
}

// ==============================================================================
// GENERATOR FUNCTION
// ==============================================================================

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

int main() {
	auto plates = generate_tectonic_plates(24, 2);
	for (auto i : plates) {
		fmt::println("TectonicPlate(vec3({}, {}, {}), {}, {}, vec3({}, {}, {})),", i.seed_dir.x, i.seed_dir.y, i.seed_dir.z, i.height, i.k, i.velocity.x, i.velocity.y, i.velocity.z);
	}
}