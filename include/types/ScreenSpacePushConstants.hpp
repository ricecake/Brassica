#pragma once

#include <cstdint>

namespace brassica {

	struct ScreenSpacePushConstants {
		std::uint32_t gPositionIndex{0};
		std::uint32_t gNormalIndex{0};
		std::uint32_t gAlbedoIndex{0};
		std::uint32_t gDepthIndex{0};
		std::uint32_t outIndirectAOIndex{0};
		std::uint32_t outShadowMaskIndex{0};

		std::uint32_t ssgiEnabled{1};
		std::uint32_t gtaoEnabled{1};
		std::uint32_t sssEnabled{1};

		float         ssgiRadius{2.0f};
		float         ssgiIntensity{1.0f};
		std::uint32_t ssgiSteps{8};
		std::uint32_t ssgiRayCount{2};

		float         gtaoRadius{2.0f};
		float         gtaoIntensity{1.0f};
		std::uint32_t gtaoSteps{4};
		std::uint32_t gtaoDirections{2};

		float         sssRadius{1.25f};
		float         sssIntensity{0.5f};
		float         sssBias{0.05f};
		std::uint32_t sssSteps{8};
	};

	static_assert(sizeof(ScreenSpacePushConstants) == 84, "ScreenSpacePushConstants must be exactly 84 bytes");

} // namespace brassica
