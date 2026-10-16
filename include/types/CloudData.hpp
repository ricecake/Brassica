#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace brassica {

	struct CloudState {
		// Resolution & Scaling
		float renderScale{1.0f};
		float worldScale{1.0f};

		// Volumetric Raymarching & Physical Properties
		float density{0.100f};
		float altitude{2000.0f};
		float thickness{1500.0f};
		float coverage{0.35f};
		float maxRayDistance{175000.0f};
		int32_t minSamples{32};
		int32_t maxSamples{96};
		float extinction{0.372f};
		glm::vec3 extinctionColor{1.0f, 1.0f, 1.0f};
		glm::vec3 albedo{0.85f, 0.85f, 0.85f};
		glm::vec3 color{1.0f, 1.0f, 1.0f};

		// Phase Function & Multiple Scattering
		float phaseG1{0.850f};
		float phaseG2{-0.200f};
		float phaseAlpha{0.500f};
		float phaseIsotropic{0.050f};
		float powderScale{0.00125f};
		float powderMultiplier{50.0f};
		float powderLocalScale{5.0f};
		float beerPowderMix{0.600f};

		// Layered Self Shadowing & Lighting
		float shadowOpticalDepthMultiplier{2.0f};
		float shadowStepMultiplier{1.0f};
		float shadowIntensity{1.0f};
		float sunLightScale{1.0f};
		float moonLightScale{1.0f};

		// Staggered Tile Scheduler
		bool enableTileScheduler{true};
		int32_t spatialUpdateFrames{16};
		float maxRefreshRate{0.25f}; // Render budget
		float priorityErrorWeight{2.5f};
		float priorityGradWeight{2.0f};
		float priorityAgeWeight{0.05f};
		float priorityNeighborErrorWeight{1.5f};
		float priorityNeighborGradWeight{1.0f};
		float priorityThreshold{0.05f};

		// SVGF Filtering & Temporal Reprojection
		bool enableTemporal{true};
		bool enableSpatialFilter{true};
		int32_t svgfPasses{2};
		float temporalGamma{1.1f};
		float maxHistoryLength{32.0f};
		float phiLuma{20.0f};
		float phiDepth{1.5f};
		float phiDensity{0.05f};
		float svgfHistoryBoost{4.0f};
		float svgfHistoryThreshold{10.0f};

		// Flow & Detail Advection
		float flowSpeed{0.25f};
		float flowDirection{3.14159265f}; // 180 degrees
		float flowHeightScale{0.015f};
		float curlStrength{10.0f};
		float curlFrequency{2.0f};
	};

	using CloudDataHost = CloudState;

} // namespace brassica
