#pragma once

#include <array>
#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	// One depth-scoped ASC CDL grading layer. Mirrors cdl_grading.glsl's CdlEntry exactly --
	// std430 buffer layout, so field order/types here must match the shader struct field-for-field.
	struct CdlEntryHost {
		glm::vec4 cdlSlope{1.0f, 1.0f, 1.0f, 0.0f};
		glm::vec4 cdlOffset{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4 cdlPower{1.0f, 1.0f, 1.0f, 0.0f};
		float     cdlSaturation{1.0f};
		float     targetDepth{0.0f};
		float     falloffWidth{1.0f};
		float     falloffRate{1.0f};
		std::int32_t priority{0};
		std::int32_t enabled{1};
		std::int32_t isMain{0};
		float     padding{0.0f};
	};

	constexpr std::size_t kMaxCdlEntries = 8;

	struct CdlGradingLayersHost {
		std::array<CdlEntryHost, kMaxCdlEntries> entries{};
		std::int32_t                             numEntries{1};
	};

	// Scene (isSky == 0) depth-based multi-layer CDL grading, uploaded to the CdlGradingLayers
	// SSBO (cdl_grading.glsl, frameSet binding 6) every frame -- read-only from the GPU's
	// perspective, so unlike ExposureDataHost there's no shader-owned state to avoid stomping.
	// Defaults to a single enabled isMain entry with an identity CDL, so behavior is unchanged
	// until something (a future settings UI) actually edits an entry.
	inline CdlGradingLayersHost s_cdlGradingLayers{
		.entries = {CdlEntryHost{.isMain = 1}},
		.numEntries = 1,
	};

} // namespace brassica
