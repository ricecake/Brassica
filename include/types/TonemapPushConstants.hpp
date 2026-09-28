#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace brassica {

	struct TonemapPushConstants {
		std::uint32_t hdrColorIndex{0};
		std::uint32_t bloomBlurIndex{0};
		std::uint32_t ltmFusedIndex{0};
		std::uint32_t ltmExpMipIndex{0};

		std::uint32_t depthTextureIndex{0};
		std::uint32_t toneMapMode{5};
		glm::vec2     ltmRes{0.0f, 0.0f};

		float intensity{0.075f};
		float minIntensity{0.05f};
		float maxIntensity{0.15f};
		float exposure{1.0f};

		float contrast{1.0f};
		float saturation{1.0f};
		float temperature{0.0f};
		float tint{0.0f};

		float pMax{1.0f};
		float pA{1.0f};
		float pM{0.22f};
		float pL{0.4f};

		float pC{1.33f};
		float pB{0.0f};
		float bloomIntensity{0.5f};
		float cdlSaturation{1.0f};

		glm::vec4 cdlSlope{1.0f, 1.0f, 1.0f, 1.0f};
		glm::vec4 cdlOffset{0.0f, 0.0f, 0.0f, 0.0f};
		glm::vec4 cdlPower{1.0f, 1.0f, 1.0f, 1.0f};

		// Appended last, deliberately: this struct has no alignas anywhere, so every vec2/vec4
		// member's offset only lines up between C++ and GLSL by the coincidence of how many 4-byte
		// scalars precede it -- inserting a new scalar field anywhere earlier shifts every field
		// after it in C++ with no compensating padding, while GLSL's std430-like push-constant
		// rules silently reinsert padding to keep vec2/vec4 8/16-byte aligned. That exact mistake
		// shipped once (bloomEnabled between toneMapMode and ltmRes): GLSL's block came out 160
		// bytes, C++'s sizeof came out 148, and vkCreateGraphicsPipelines correctly rejected it.
		// Appending after the last member (a vec4, always ending on a 16-byte boundary either way)
		// needs no padding in either language, so this is the only structurally safe place to add a
		// new scalar without either an explicit alignas or auditing every offset by hand again.
		std::int32_t bloomEnabled{1};
	};

	static_assert(sizeof(TonemapPushConstants) == 148, "TonemapPushConstants size must be 148 bytes -- must match tonemap.frag's push_constant block exactly");

	inline TonemapPushConstants s_tonemapPush{};

	// Master switch for TonemapComputeNode's entire dispatch (bloom downsample + LTM fuse +
	// auto-exposure histogram accumulation, all interleaved in bloom_downsample.comp -- they can't
	// be split apart without shader surgery, so this is an all-or-nothing pass toggle). Distinct
	// from TonemapPushConstants::bloomEnabled above, which only hides bloom's contribution in the
	// final composite and works independently of whether this pass ran. Checked in Setup(), not a
	// push-constant field, since it needs to gate Recipe::isActive before any dispatch happens.
	inline bool s_tonemapComputePassEnabled{true};

} // namespace brassica
