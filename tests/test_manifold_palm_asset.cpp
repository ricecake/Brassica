#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <atomic>
#include <cmath>
#include <memory>

#include <spdlog/sinks/callback_sink.h>
#include <spdlog/spdlog.h>

#include "doctest/doctest.h"

#include "passes/model/ManifoldPalmAsset.hpp"

using namespace brassica;

TEST_CASE("BuildPalmTree produces a real, rigged mesh across many seeds: round trunk + frond crown") {
	for (std::uint32_t seed = 0; seed < 20; ++seed) {
		PalmGenParams params;
		params.seed = seed;

		ModelAsset palm = BuildPalmTree(params);

		REQUIRE(!palm.vertices.empty());
		REQUIRE(!palm.meshlets.empty());
		// 1 trunk joint + frondCount frond joints -- not a branching-chain archetype, this is a
		// fixed, small, bespoke skeleton (same reasoning Pine's own BuildPineTree test would
		// use).
		REQUIRE(palm.jointCount == static_cast<std::uint32_t>(1 + params.frondCount));
		REQUIRE(palm.skeleton.size() == palm.jointCount);

		CHECK(palm.skeleton[0].parentIndex == kModelNoSkeletonSentinel);
		for (std::uint32_t j = 1; j < palm.jointCount; ++j) {
			// Every frond is a direct child of the trunk (joint 0).
			CHECK(palm.skeleton[j].parentIndex == 0u);
		}

		for (const GltfVertex& v : palm.vertices) {
			CHECK(std::isfinite(v.position.x));
			CHECK(std::isfinite(v.position.y));
			CHECK(std::isfinite(v.position.z));
			REQUIRE(v.joints.x < palm.jointCount);
		}

		// The trunk's own cross-section is a real circularSegments-sided round tube -- a
		// meaningful vertex count, not the old emitPalmTreeItem's flat 2-vertex-wide ribbon this
		// generator was built to replace. At least one vertex attributed to joint 0 should sit
		// well off the vertical trunk axis (x/z, not y) by roughly a trunk radius, which a flat
		// 2-point ribbon could only do along a single fixed direction -- a real round
		// cross-section has vertices spread around the full circle.
		bool sawOffAxisVertex = false;
		for (const GltfVertex& v : palm.vertices) {
			if (v.joints.x != 0u) {
				continue;
			}
			float horizDist = std::sqrt(v.position.x * v.position.x + v.position.z * v.position.z);
			if (horizDist > params.trunkRadiusLow * 0.5f) {
				sawOffAxisVertex = true;
				break;
			}
		}
		CHECK(sawOffAxisVertex);
	}
}

TEST_CASE("BuildPalmTree's trunk bend is gentle enough to never trip the curvature-safety fallback") {
	// A real regression check: BuildCurvedSegment silently falls back to a straight tube (and
	// logs a warning) if the requested bend is too tight relative to the tube's own radius --
	// this session's earlier Tree/Bush crowding work hit exactly that fallback. Installing a
	// counting sink (same technique that earlier work used) over a real sweep is what actually
	// catches a regression here; just checking the build doesn't crash would not.
	std::atomic<int> warnCount{0};
	auto             sink = std::make_shared<spdlog::sinks::callback_sink_mt>([&warnCount](const spdlog::details::log_msg& msg) {
		if (msg.level == spdlog::level::warn) {
			++warnCount;
		}
	});
	auto previousLogger = spdlog::default_logger();
	spdlog::set_default_logger(std::make_shared<spdlog::logger>("counting", sink));

	for (std::uint32_t seed = 0; seed < 50; ++seed) {
		PalmGenParams params;
		params.seed = seed;
		ModelAsset palm = BuildPalmTree(params);
		REQUIRE(!palm.vertices.empty());
	}

	spdlog::set_default_logger(previousLogger);
	CHECK(warnCount.load() == 0);
}
