#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	// Mirrors shaders/atmosphere/composite.frag's push_constant block. Per-ray values only (sun/
	// moon direction+radiance, step count, bindless indices) -- tuning constants come from the
	// global AtmosphereUBO (atmosphere/common.glsl, set 0 binding 4), not from here, same
	// separation NodeFrameParams already draws between "changes every frame" and "read from the
	// UBO." Uses the vec3-then-scalar pairing AtmospherePushConstants already establishes (a
	// trailing alignas(4) scalar packs into the vec3's own alignas(16) padding, matching
	// std430's push-constant layout on the GLSL side -- no explicit layout(offset=) needed since
	// nothing outside this struct depends on a specific field's byte offset).
	struct AtmosphereCompositePushConstants {
		alignas(16) glm::vec3 sunDir{0.0f, 1.0f, 0.0f};
		alignas(4) float maxUnderwaterDistanceKM{0.2f};
		alignas(16) glm::vec3 sunRadiance{10.0f, 9.5f, 8.5f};
		alignas(4) float multiScatScale{1.0f};
		alignas(16) glm::vec3 moonDir{0.0f, -1.0f, 0.0f};
		alignas(4) std::uint32_t stepCount{16};
		alignas(16) glm::vec3 moonRadiance{0.1f, 0.12f, 0.16f};
		alignas(4) std::uint32_t gPositionIndex{0};
		alignas(4) std::uint32_t gAlbedoIndex{0};
		alignas(4) std::uint32_t hdrColorIndex{0};
		alignas(4) std::uint32_t transmittanceIndex{0};
		alignas(4) std::uint32_t multiScatteringIndex{0};
	};

} // namespace brassica
