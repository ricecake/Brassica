#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>

#include "doctest/doctest.h"

#include "types/AtmospherePushConstants.hpp"

// evaluateAerialPerspective (and its replacement, evaluateAtmosphere) integrate extinction in
// KILOMETRES (dt = distanceKM / steps, exp(-extinction * dt)), but water.frag's own
// already-correct surface-crossing math uses per-METRE coefficients for the same physical
// quantity ({0.28, 0.07, 0.02}). AtmospherePushConstants::waterExtinctionBase/waterScatteringBase
// were calibrated as if they were also per-metre, making them ~1000-2300x too weak once actually
// integrated in km -- the reported symptom ("water just makes things darker, no color shift") is
// this unit bug, not a missing effect. This test proves the fix numerically against the real
// struct before any shader/node code exists: run once against the unmodified struct to confirm it
// fails the way the bug predicts, then apply the coefficient fix and confirm it passes.

TEST_CASE("AtmospherePushConstants water coefficients produce a real wavelength-dependent color shift") {
	brassica::AtmospherePushConstants atmosphere{};

	auto transmittanceAtDepthMeters = [&](float depthMeters) {
		float depthKM = depthMeters / 1000.0f;
		return glm::vec3{
			std::exp(-atmosphere.waterExtinctionBase.x * atmosphere.waterScale * depthKM),
			std::exp(-atmosphere.waterExtinctionBase.y * atmosphere.waterScale * depthKM),
			std::exp(-atmosphere.waterExtinctionBase.z * atmosphere.waterScale * depthKM),
		};
	};

	SUBCASE("10m depth: real color shift, red absorbed far faster than blue") {
		glm::vec3 t = transmittanceAtDepthMeters(10.0f);
		CHECK(t.r < doctest::Approx(0.10));
		CHECK(t.b > doctest::Approx(0.60));
		CHECK(t.b / t.r > 5.0f);
	}

	SUBCASE("2m depth: shallow submersion is already visibly tinted") {
		glm::vec3 t = transmittanceAtDepthMeters(2.0f);
		CHECK(t.b / t.r > 1.4f);
	}

	SUBCASE("single-scatter albedo matches water.frag's own shallowWaterTint") {
		// water.frag: shallowWaterTint = vec3(0.12, 0.62, 0.78) -- the color deep-ish water tends
		// toward. sigma_s/sigma_t (scattering/extinction) is that same ratio by construction, so
		// the new composite pass and water.frag's existing surface math agree on what water looks
		// like, rather than each inventing its own answer.
		glm::vec3 albedo{
			atmosphere.waterScatteringBase.x / atmosphere.waterExtinctionBase.x,
			atmosphere.waterScatteringBase.y / atmosphere.waterExtinctionBase.y,
			atmosphere.waterScatteringBase.z / atmosphere.waterExtinctionBase.z,
		};
		CHECK(albedo.r == doctest::Approx(0.12).epsilon(0.02));
		CHECK(albedo.g == doctest::Approx(0.62).epsilon(0.02));
		CHECK(albedo.b == doctest::Approx(0.78).epsilon(0.02));
		CHECK(albedo.r < albedo.g);
		CHECK(albedo.g < albedo.b);
	}
}
