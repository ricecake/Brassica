#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <cmath>
#include <glm/glm.hpp>
#include "doctest/doctest.h"
#include "types/ubo/CascadedShadowUBO.hpp"
#include "EngineConstants.hpp"

TEST_CASE("CascadedShadowUBO layout and alignment requirements") {
	CHECK(sizeof(brassica::CascadedShadowUBO) == 304);
	CHECK(alignof(brassica::CascadedShadowUBO) == 16);

	brassica::CascadedShadowUBO ubo{};
	CHECK(ubo.cascadeSplits.x == 50.0f);
	CHECK(ubo.cascadeSplits.y == 200.0f);
	CHECK(ubo.cascadeSplits.z == 800.0f);
	CHECK(ubo.cascadeSplits.w == 3200.0f);
	CHECK(ubo.numCascades == 4);
}

TEST_CASE("Cascaded shadow refresh rate throttling bitmask logic") {
	auto getRefreshMask = [](std::uint64_t frameIndex) -> std::uint32_t {
		std::uint32_t mask = 0;
		if (frameIndex % 1u == 0u) mask |= (1u << 0);
		if (frameIndex % 2u == 0u) mask |= (1u << 1);
		if (frameIndex % 4u == 0u) mask |= (1u << 2);
		if (frameIndex % 8u == 0u) mask |= (1u << 3);
		return mask;
	};

	// Frame 0: all cascades active
	CHECK(getRefreshMask(0) == 0xF);
	// Frame 1: only Cascade 0 active
	CHECK(getRefreshMask(1) == 0x1);
	// Frame 2: Cascade 0 and Cascade 1 active
	CHECK(getRefreshMask(2) == 0x3);
	// Frame 3: only Cascade 0 active
	CHECK(getRefreshMask(3) == 0x1);
	// Frame 4: Cascade 0, 1, 2 active
	CHECK(getRefreshMask(4) == 0x7);
	// Frame 8: all cascades active
	CHECK(getRefreshMask(8) == 0xF);
}

TEST_CASE("Planet horizon and curvature shadow logic") {
	float planetRadius = brassica::FAKE_PLANET_RADIUS; // 600,000m
	glm::vec3 planetCenter(0.0f, -planetRadius, 0.0f);

	// Point on top of planet surface at (0, 0, 0)
	glm::vec3 surfacePoint(0.0f, 0.0f, 0.0f);
	glm::vec3 fragToCenter = surfacePoint - planetCenter;
	glm::vec3 planetNormal = glm::normalize(fragToCenter);

	// Sun directly overhead (0, 1, 0) -> unshadowed
	glm::vec3 sunOverhead(0.0f, 1.0f, 0.0f);
	float NdotL_overhead = glm::dot(planetNormal, sunOverhead);
	CHECK(NdotL_overhead > 0.99f);

	// Sun below horizon (0, -1, 0) -> fully shadowed
	glm::vec3 sunNight(0.0f, -1.0f, 0.0f);
	float NdotL_night = glm::dot(planetNormal, sunNight);
	CHECK(NdotL_night < -0.99f);
}
